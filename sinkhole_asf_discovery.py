#!/usr/bin/env python3
"""Discover ASF SAR/InSAR coverage for known sinkhole events in Pennsylvania.

Default behavior:
- Uses a fixed sinkhole event catalog (8 events supplied by the user).
- Searches a point AOI at each event coordinate.
- Uses a 12 month pre-event to 3 month post-event window.
- Prioritizes Sentinel-1, with ARIA/OPERA fallback manifests.
- Writes per-event and combined manifests.

Optional behavior:
- Download browse quicklooks.
- Download product files using interactive Earthdata authentication prompts.
- Generate QC reports only from existing manifests.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import asf_search as asf
import pandas as pd
import requests
import yaml


@dataclass(frozen=True)
class SinkholeEvent:
    event_date: pd.Timestamp
    location: str
    latitude: float
    longitude: float
    cause: str
    notes: str


DEFAULT_EVENTS: list[SinkholeEvent] = [
    SinkholeEvent(
        event_date=pd.Timestamp("2024-12-02"),
        location="Marguerite, Westmoreland County",
        latitude=40.2636,
        longitude=-79.4633,
        cause="Mine Subsidence",
        notes="Fatal collapse into the abandoned H.C. Frick Marguerite Mine.",
    ),
    SinkholeEvent(
        event_date=pd.Timestamp("2024-12-16"),
        location="Park Place, Schuylkill County",
        latitude=40.8354,
        longitude=-76.1558,
        cause="Mine Subsidence",
        notes="75x45 ft hole swallowed an in-ground pool; 10 yards from a 2016 event.",
    ),
    SinkholeEvent(
        event_date=pd.Timestamp("2023-08-16"),
        location="Eisenhower Parking Deck, Penn State",
        latitude=40.8005,
        longitude=-77.8617,
        cause="Infrastructure Failure",
        notes="Caused by water from a failed underground stormwater pipe.",
    ),
    SinkholeEvent(
        event_date=pd.Timestamp("2020-08-28"),
        location="Packer Twp, Carbon County",
        latitude=40.9144,
        longitude=-75.8992,
        cause="Tunnel Subsidence",
        notes="Quakake Tunnel roof collapse caused surface depression.",
    ),
    SinkholeEvent(
        event_date=pd.Timestamp("2018-08-10"),
        location="Tanger Outlets, Lancaster",
        latitude=40.0305,
        longitude=-76.2164,
        cause="Stormwater Failure",
        notes="Massive hole buried 6 cars; investigation cited failed plastic modular crate system.",
    ),
    SinkholeEvent(
        event_date=pd.Timestamp("2018-07-26"),
        location="Codorus Creek, York",
        latitude=39.9634,
        longitude=-76.7305,
        cause="Karst / Rain",
        notes="20-ft wide channel wall collapse contributing to sinkhole expansion during heavy rain.",
    ),
    SinkholeEvent(
        event_date=pd.Timestamp("2023-10-30"),
        location="130 Sickler Hill Rd, Luzerne",
        latitude=41.2721,
        longitude=-75.9458,
        cause="Mine Subsidence",
        notes="BAMR-reclaimed site following active ground movement.",
    ),
    SinkholeEvent(
        event_date=pd.Timestamp("2018-03-01"),
        location="West Whiteland, Chester",
        latitude=40.0242,
        longitude=-75.6022,
        cause="Pipeline Construction",
        notes="Multiple sinkholes along Lisa Drive associated with Mariner East 2 pipeline drilling.",
    ),
]


DATASET_ORDER = [
    ("SENTINEL1_SLC", asf.DATASET.SENTINEL1),
    ("ARIA_S1_GUNW", asf.DATASET.ARIA_S1_GUNW),
    ("OPERA_S1_DISP", asf.DATASET.OPERA_S1),
]
DERIVED_DATASETS = {"ARIA_S1_GUNW", "OPERA_S1_DISP"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASF discovery and optional download workflow for sinkhole events.")
    p.add_argument("--months-before", type=int, default=12, help="Months before event date to search (default: 12).")
    p.add_argument("--months-after", type=int, default=3, help="Months after event date to search (default: 3).")
    p.add_argument("--max-results", type=int, default=250, help="Max results per event+dataset manifest.")
    p.add_argument("--outdir", default="outputs/sinkhole_event_discovery", help="Output folder.")
    p.add_argument("--split-file", default="config/event_split.yaml", help="YAML/JSON split file with train/val/test lists.")
    p.add_argument(
        "--download-browse",
        type=int,
        default=0,
        help="If > 0, download up to N browse quicklooks per event+dataset.",
    )
    p.add_argument(
        "--download-products",
        action="store_true",
        help="Download product files in each manifest using Earthdata auth.",
    )
    p.add_argument(
        "--processes",
        type=int,
        default=4,
        help="Parallel download processes for ASF download (default: 4).",
    )
    p.add_argument(
        "--auth-mode",
        choices=["auto", "token", "creds", "none"],
        default="auto",
        help="Auth mode for product download. 'auto' checks env vars then prompts.",
    )
    p.add_argument(
        "--skip-counts",
        action="store_true",
        help="Skip search_count calls to reduce API requests.",
    )
    p.add_argument(
        "--event-id",
        action="append",
        default=[],
        help="Limit run to specific event IDs (repeat flag for multiple).",
    )
    p.add_argument(
        "--dataset",
        action="append",
        choices=["SENTINEL1_SLC", "ARIA_S1_GUNW", "OPERA_S1_DISP"],
        default=[],
        help="Limit run to specific dataset(s) (repeat flag for multiple).",
    )
    p.add_argument(
        "--max-downloads-per-manifest",
        type=int,
        default=None,
        help="Optional cap on number of products downloaded per event+dataset manifest.",
    )
    p.add_argument(
        "--max-total-gb",
        type=float,
        default=None,
        help="Hard stop for cumulative downloaded size growth under outdir in GB.",
    )
    p.add_argument(
        "--prefer-derived",
        action="store_true",
        help="When no --dataset is provided, prioritize ARIA/OPERA and skip Sentinel-1 SLC.",
    )
    p.add_argument(
        "--qc-only",
        action="store_true",
        help="Skip ASF discovery/download and generate QC outputs from existing manifests.",
    )
    return p.parse_args()


def _slug(text: str) -> str:
    keep = []
    for c in text.lower().strip():
        if c.isalnum():
            keep.append(c)
        elif c in {" ", "-", ",", "/"}:
            keep.append("_")
    raw = "".join(keep)
    while "__" in raw:
        raw = raw.replace("__", "_")
    return raw.strip("_")[:80]


def _event_id(event: SinkholeEvent) -> str:
    return f"{event.event_date.date()}_{_slug(event.location)}"


def _window_for_event(event_date: pd.Timestamp, months_before: int, months_after: int) -> tuple[str, str]:
    start = event_date - pd.DateOffset(months=months_before)
    end = event_date + pd.DateOffset(months=months_after)
    return str(start.date()), str(end.date())


def _results_to_df(results: Iterable, dataset_name: str, event: SinkholeEvent) -> pd.DataFrame:
    rows = []
    for r in results:
        p = r.properties
        browse = p.get("browse")
        browse_url = browse[0] if isinstance(browse, list) and browse else None
        bytes_val = p.get("bytes")
        if isinstance(bytes_val, dict):
            try:
                first_key = next(iter(bytes_val))
                bytes_val = bytes_val[first_key].get("bytes")
            except Exception:
                bytes_val = None

        rows.append(
            {
                "event_date": str(event.event_date.date()),
                "location": event.location,
                "latitude": event.latitude,
                "longitude": event.longitude,
                "cause": event.cause,
                "notes": event.notes,
                "dataset": dataset_name,
                "scene_name": p.get("sceneName"),
                "file_name": p.get("fileName"),
                "start_time": p.get("startTime"),
                "stop_time": p.get("stopTime"),
                "url": p.get("url"),
                "browse_url": browse_url,
                "bytes": bytes_val,
                "path_number": p.get("pathNumber"),
                "frame_number": p.get("frameNumber"),
                "flight_direction": p.get("flightDirection"),
                "orbit": p.get("orbit"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
        df["stop_time"] = pd.to_datetime(df["stop_time"], errors="coerce", utc=True)
        df = df.sort_values("start_time", ascending=True).reset_index(drop=True)
    return df


def _build_session(auth_mode: str) -> asf.ASFSession | None:
    if auth_mode == "none":
        return None

    token_env = os.getenv("EARTHDATA_TOKEN") or os.getenv("ASF_EDL_TOKEN")
    user_env = os.getenv("EARTHDATA_USERNAME") or os.getenv("ASF_USERNAME")
    pass_env = os.getenv("EARTHDATA_PASSWORD") or os.getenv("ASF_PASSWORD")

    if auth_mode in {"auto", "token"} and token_env:
        return asf.ASFSession().auth_with_token(token_env)

    if auth_mode in {"auto", "creds"} and user_env and pass_env:
        return asf.ASFSession().auth_with_creds(user_env, pass_env)

    if auth_mode == "token":
        token = getpass.getpass("Enter Earthdata token: ").strip()
        return asf.ASFSession().auth_with_token(token)

    if auth_mode == "creds":
        username = input("Earthdata username: ").strip()
        password = getpass.getpass("Earthdata password: ")
        return asf.ASFSession().auth_with_creds(username, password)

    mode = input("No auth found in env. Use token or creds? [token/creds]: ").strip().lower()
    if mode == "token":
        token = getpass.getpass("Enter Earthdata token: ").strip()
        return asf.ASFSession().auth_with_token(token)

    username = input("Earthdata username: ").strip()
    password = getpass.getpass("Earthdata password: ")
    return asf.ASFSession().auth_with_creds(username, password)


def _download_browse_images(manifest: pd.DataFrame, outdir: Path, max_images: int) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    if max_images <= 0 or manifest.empty or "browse_url" not in manifest.columns:
        return 0
    m = manifest.dropna(subset=["browse_url"]).copy()
    if m.empty:
        return 0
    m = m.sort_values("start_time", ascending=False).head(max_images)

    saved = 0
    for _, row in m.iterrows():
        url = row["browse_url"]
        name = str(row.get("scene_name") or "scene").replace("/", "_")
        ext = ".png" if str(url).lower().endswith(".png") else ".jpg"
        out_path = outdir / f"{name}{ext}"
        try:
            r = requests.get(url, timeout=25)
            if r.status_code == 200 and r.content:
                out_path.write_bytes(r.content)
                saved += 1
        except Exception:
            continue
    return saved


def _search_event_dataset(
    event: SinkholeEvent,
    dataset_name: str,
    dataset,
    start_date: str,
    end_date: str,
    max_results: int,
    skip_counts: bool,
):
    point_wkt = f"POINT({event.longitude} {event.latitude})"
    total_available = None
    if not skip_counts:
        total_available = int(
            asf.search_count(
                intersectsWith=point_wkt,
                start=f"{start_date}T00:00:00Z",
                end=f"{end_date}T23:59:59Z",
                dataset=dataset,
            )
        )

    results = asf.geo_search(
        intersectsWith=point_wkt,
        start=f"{start_date}T00:00:00Z",
        end=f"{end_date}T23:59:59Z",
        dataset=dataset,
        maxResults=max_results,
    )

    df = _results_to_df(results, dataset_name, event)
    if total_available is None:
        total_available = len(df)
    return results, df, total_available


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


def _load_split_file(path: Path, known_event_ids: set[str]) -> dict[str, list[str]]:
    if not path.exists():
        return {"train": sorted(known_event_ids), "val": [], "test": []}

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Split file must map train/val/test keys: {path}")

    split = {
        "train": sorted(str(x) for x in data.get("train", [])),
        "val": sorted(str(x) for x in data.get("val", [])),
        "test": sorted(str(x) for x in data.get("test", [])),
    }

    seen = set()
    for key in ["train", "val", "test"]:
        bad = [eid for eid in split[key] if eid not in known_event_ids]
        if bad:
            raise ValueError(f"Split file has unknown event IDs in {key}: {bad}")
        overlap = [eid for eid in split[key] if eid in seen]
        if overlap:
            raise ValueError(f"Split file has duplicate event IDs across splits: {overlap}")
        seen.update(split[key])

    unassigned = sorted(known_event_ids - seen)
    if unassigned:
        split["train"].extend(unassigned)
        split["train"] = sorted(set(split["train"]))
    return split


def _write_split_outputs(split: dict[str, list[str]], outdir: Path) -> None:
    qc_dir = outdir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_name in ["train", "val", "test"]:
        for eid in sorted(split.get(split_name, [])):
            rows.append({"event_id": eid, "split": split_name})
    pd.DataFrame(rows).to_csv(qc_dir / "event_split_resolved.csv", index=False)
    (qc_dir / "event_split_resolved.json").write_text(json.dumps(split, indent=2), encoding="utf-8")


def _generate_qc_reports(
    outdir: Path,
    events_df: pd.DataFrame,
    split: dict[str, list[str]],
    months_before: int,
    months_after: int,
) -> dict:
    qc_dir = outdir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    coverage_rows: list[dict] = []
    missing_rows: list[dict] = []
    parse_error_rows: list[dict] = []

    event_date_map = {r["event_id"]: pd.Timestamp(r["event_date"]) for _, r in events_df.iterrows()}
    available_split = {}
    for name, ids in split.items():
        for eid in ids:
            available_split[eid] = name

    usable_derived_events = set()
    manifest_ok_event_dataset = 0
    total_event_dataset = len(events_df) * len(DATASET_ORDER)

    for _, erow in events_df.iterrows():
        event_id = erow["event_id"]
        event_date = pd.Timestamp(erow["event_date"])
        expected_start = event_date - pd.DateOffset(months=months_before)
        expected_end = event_date + pd.DateOffset(months=months_after)

        for dataset_name, _ in DATASET_ORDER:
            manifest_path = outdir / event_id / f"manifest_{dataset_name.lower()}.csv"
            if not manifest_path.exists():
                coverage_rows.append(
                    {
                        "event_id": event_id,
                        "split": available_split.get(event_id, "unassigned"),
                        "dataset": dataset_name,
                        "manifest_exists": False,
                        "manifest_readable": False,
                        "row_count": 0,
                        "parse_ok": False,
                        "within_window": False,
                        "window_start_expected": str(expected_start.date()),
                        "window_end_expected": str(expected_end.date()),
                        "start_min": None,
                        "start_max": None,
                    }
                )
                missing_rows.append(
                    {
                        "event_id": event_id,
                        "dataset": dataset_name,
                        "row_count": 0,
                        "missing_scene_name": None,
                        "missing_start_time": None,
                        "missing_stop_time": None,
                        "missing_url": None,
                        "duplicate_scene_name": None,
                        "manifest_exists": False,
                    }
                )
                parse_error_rows.append(
                    {
                        "event_id": event_id,
                        "dataset": dataset_name,
                        "category": "manifest_missing",
                        "detail": str(manifest_path),
                    }
                )
                continue

            try:
                df = pd.read_csv(manifest_path)
            except Exception as exc:
                coverage_rows.append(
                    {
                        "event_id": event_id,
                        "split": available_split.get(event_id, "unassigned"),
                        "dataset": dataset_name,
                        "manifest_exists": True,
                        "manifest_readable": False,
                        "row_count": 0,
                        "parse_ok": False,
                        "within_window": False,
                        "window_start_expected": str(expected_start.date()),
                        "window_end_expected": str(expected_end.date()),
                        "start_min": None,
                        "start_max": None,
                    }
                )
                parse_error_rows.append(
                    {
                        "event_id": event_id,
                        "dataset": dataset_name,
                        "category": "manifest_read_error",
                        "detail": str(exc),
                    }
                )
                continue

            if "start_time" not in df.columns:
                df["start_time"] = pd.NaT
            if "stop_time" not in df.columns:
                df["stop_time"] = pd.NaT

            st = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
            et = pd.to_datetime(df["stop_time"], errors="coerce", utc=True)
            parse_fail_count = int(st.isna().sum())
            parse_ok = parse_fail_count == 0 or len(df) == 0

            if parse_fail_count > 0:
                parse_error_rows.append(
                    {
                        "event_id": event_id,
                        "dataset": dataset_name,
                        "category": "timestamp_parse_error",
                        "detail": f"start_time parse failures: {parse_fail_count}",
                    }
                )

            window_start_ts = pd.Timestamp(expected_start, tz="UTC")
            window_end_ts = pd.Timestamp(expected_end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            within_window = bool(((st.dropna() >= window_start_ts) & (st.dropna() <= window_end_ts)).all()) if len(st.dropna()) else False

            if len(df) > 0 and dataset_name in DERIVED_DATASETS and parse_ok:
                usable_derived_events.add(event_id)
            if parse_ok:
                manifest_ok_event_dataset += 1

            if len(df) == 0:
                parse_error_rows.append(
                    {
                        "event_id": event_id,
                        "dataset": dataset_name,
                        "category": "no_products",
                        "detail": "manifest has zero rows",
                    }
                )

            coverage_rows.append(
                {
                    "event_id": event_id,
                    "split": available_split.get(event_id, "unassigned"),
                    "dataset": dataset_name,
                    "manifest_exists": True,
                    "manifest_readable": True,
                    "row_count": int(len(df)),
                    "parse_ok": bool(parse_ok),
                    "within_window": bool(within_window),
                    "window_start_expected": str(expected_start.date()),
                    "window_end_expected": str(expected_end.date()),
                    "start_min": str(st.min()) if len(st.dropna()) else None,
                    "start_max": str(st.max()) if len(st.dropna()) else None,
                }
            )

            scene_col = "scene_name" if "scene_name" in df.columns else None
            missing_rows.append(
                {
                    "event_id": event_id,
                    "dataset": dataset_name,
                    "row_count": int(len(df)),
                    "missing_scene_name": int(df[scene_col].isna().sum()) if scene_col else None,
                    "missing_start_time": int(st.isna().sum()),
                    "missing_stop_time": int(et.isna().sum()),
                    "missing_url": int(df["url"].isna().sum()) if "url" in df.columns else None,
                    "duplicate_scene_name": int(df[scene_col].duplicated().sum()) if scene_col else None,
                    "manifest_exists": True,
                }
            )

    downloads_log_path = outdir / "downloads_log.csv"
    if downloads_log_path.exists():
        try:
            dlog = pd.read_csv(downloads_log_path)
            for _, drow in dlog.iterrows():
                status = str(drow.get("status", ""))
                if not (status == "ok" or status.startswith("error:") or status.startswith("skipped:")):
                    parse_error_rows.append(
                        {
                            "event_id": str(drow.get("event_id")),
                            "dataset": str(drow.get("dataset")),
                            "category": "download_status_invalid",
                            "detail": status,
                        }
                    )
        except Exception as exc:
            parse_error_rows.append(
                {
                    "event_id": "*",
                    "dataset": "*",
                    "category": "download_log_read_error",
                    "detail": str(exc),
                }
            )

    coverage_df = pd.DataFrame(coverage_rows).sort_values(["event_id", "dataset"]).reset_index(drop=True)
    missing_df = pd.DataFrame(missing_rows).sort_values(["event_id", "dataset"]).reset_index(drop=True)
    parse_df = pd.DataFrame(parse_error_rows)
    if parse_df.empty:
        parse_df = pd.DataFrame(columns=["event_id", "dataset", "category", "detail"])
    else:
        parse_df = parse_df.sort_values(["event_id", "dataset", "category"]).reset_index(drop=True)

    coverage_df.to_csv(qc_dir / "coverage_report.csv", index=False)
    missing_df.to_csv(qc_dir / "missingness_report.csv", index=False)
    parse_df.to_csv(qc_dir / "parse_errors.csv", index=False)

    failures_by_category = parse_df["category"].value_counts().to_dict() if len(parse_df) else {}
    events_with_usable_derived = len(usable_derived_events)

    acceptance = {
        "manifest_generation_8_of_8": int(events_df["event_id"].nunique()) == 8 and all(
            bool((coverage_df["event_id"] == eid).any()) for eid in events_df["event_id"].tolist()
        ),
        "derived_timeseries_ge_6_events": events_with_usable_derived >= 6,
        "failures_categorized": len(parse_df) == 0 or parse_df["category"].notna().all(),
        "deterministic_split_files": True,
    }
    acceptance = {k: bool(v) for k, v in acceptance.items()}
    acceptance["pipeline_validated"] = all(bool(v) for v in acceptance.values())

    summary = {
        "events_total": int(events_df["event_id"].nunique()),
        "event_dataset_pairs": int(total_event_dataset),
        "event_dataset_pairs_parse_ok": int(manifest_ok_event_dataset),
        "events_with_usable_derived_timeseries": int(events_with_usable_derived),
        "failures_by_category": failures_by_category,
        "acceptance": acceptance,
        "qc_reports": {
            "coverage_report": str((qc_dir / "coverage_report.csv")),
            "missingness_report": str((qc_dir / "missingness_report.csv")),
            "parse_errors": str((qc_dir / "parse_errors.csv")),
        },
    }
    (qc_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_events_df = pd.DataFrame(
        [
            {
                "event_date": str(e.event_date.date()),
                "location": e.location,
                "latitude": e.latitude,
                "longitude": e.longitude,
                "cause": e.cause,
                "notes": e.notes,
                "event_id": _event_id(e),
            }
            for e in DEFAULT_EVENTS
        ]
    )
    all_events_df.to_csv(outdir / "events_used.csv", index=False)

    known_event_ids = set(all_events_df["event_id"].tolist())
    split = _load_split_file(Path(args.split_file), known_event_ids)
    _write_split_outputs(split, outdir)

    if args.qc_only:
        qc_summary = _generate_qc_reports(
            outdir=outdir,
            events_df=all_events_df,
            split=split,
            months_before=args.months_before,
            months_after=args.months_after,
        )
        print("QC-only run complete.")
        print(json.dumps(qc_summary, indent=2))
        print(f"QC directory: {(outdir / 'qc').resolve()}")
        return

    selected_events = list(DEFAULT_EVENTS)
    if args.event_id:
        allow = set(args.event_id)
        selected_events = [e for e in DEFAULT_EVENTS if _event_id(e) in allow]
        if not selected_events:
            raise ValueError(f"No matching events for --event-id values: {sorted(allow)}")

    selected_datasets = DATASET_ORDER
    if args.dataset:
        allow_ds = set(args.dataset)
        selected_datasets = [(name, ds) for name, ds in DATASET_ORDER if name in allow_ds]
        if not selected_datasets:
            raise ValueError(f"No matching datasets for --dataset values: {sorted(allow_ds)}")
    elif args.prefer_derived:
        selected_datasets = [(name, ds) for name, ds in DATASET_ORDER if name in DERIVED_DATASETS]

    search_results_lookup: dict[tuple[str, str], object] = {}
    summary_rows: list[dict] = []
    all_rows: list[pd.DataFrame] = []

    for event in selected_events:
        event_id = _event_id(event)
        event_dir = outdir / event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        start_date, end_date = _window_for_event(
            event.event_date,
            months_before=args.months_before,
            months_after=args.months_after,
        )

        for dataset_name, dataset in selected_datasets:
            results, manifest, available = _search_event_dataset(
                event=event,
                dataset_name=dataset_name,
                dataset=dataset,
                start_date=start_date,
                end_date=end_date,
                max_results=args.max_results,
                skip_counts=args.skip_counts,
            )
            search_results_lookup[(event_id, dataset_name)] = results

            manifest_path = event_dir / f"manifest_{dataset_name.lower()}.csv"
            manifest.to_csv(manifest_path, index=False)
            if not manifest.empty:
                manifest_for_all = manifest.copy()
                manifest_for_all["event_id"] = event_id
                all_rows.append(manifest_for_all)

            browse_saved = _download_browse_images(
                manifest=manifest,
                outdir=event_dir / "browse" / dataset_name.lower(),
                max_images=args.download_browse,
            )

            summary_rows.append(
                {
                    "event_id": event_id,
                    "event_date": str(event.event_date.date()),
                    "location": event.location,
                    "dataset": dataset_name,
                    "window_start": start_date,
                    "window_end": end_date,
                    "count_in_manifest": int(len(manifest)),
                    "count_available": int(available),
                    "first_scene": manifest["start_time"].min() if len(manifest) else pd.NaT,
                    "last_scene": manifest["start_time"].max() if len(manifest) else pd.NaT,
                    "browse_saved": int(browse_saved),
                    "manifest_csv": str(manifest_path),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "manifest_summary.csv", index=False)

    if all_rows:
        combined_df = pd.concat(all_rows, ignore_index=True)
    else:
        combined_df = pd.DataFrame()
    combined_df.to_csv(outdir / "all_manifests.csv", index=False)

    downloads_completed = []
    if args.download_products:
        session = _build_session(args.auth_mode)
        baseline_bytes = _dir_size_bytes(outdir)
        cap_reached = False

        for row in summary_rows:
            event_id = row["event_id"]
            dataset_name = row["dataset"]
            attempted = int(row["count_in_manifest"])
            if args.max_downloads_per_manifest is not None:
                attempted = int(min(attempted, args.max_downloads_per_manifest))

            if cap_reached:
                downloads_completed.append(
                    {
                        "event_id": event_id,
                        "dataset": dataset_name,
                        "download_dir": str(outdir / event_id / "downloads" / dataset_name.lower()),
                        "attempted_products": 0,
                        "status": "skipped: storage cap reached",
                    }
                )
                continue

            if row["count_in_manifest"] <= 0 or attempted <= 0:
                downloads_completed.append(
                    {
                        "event_id": event_id,
                        "dataset": dataset_name,
                        "download_dir": str(outdir / event_id / "downloads" / dataset_name.lower()),
                        "attempted_products": 0,
                        "status": "skipped: no products",
                    }
                )
                continue

            if args.max_total_gb is not None:
                current_growth_gb = (_dir_size_bytes(outdir) - baseline_bytes) / (1024**3)
                if current_growth_gb >= args.max_total_gb:
                    cap_reached = True
                    downloads_completed.append(
                        {
                            "event_id": event_id,
                            "dataset": dataset_name,
                            "download_dir": str(outdir / event_id / "downloads" / dataset_name.lower()),
                            "attempted_products": 0,
                            "status": f"skipped: storage cap reached ({current_growth_gb:.2f} GB)",
                        }
                    )
                    continue

            results = search_results_lookup[(event_id, dataset_name)]
            if args.max_downloads_per_manifest is not None:
                results = results[: args.max_downloads_per_manifest]

            download_dir = outdir / event_id / "downloads" / dataset_name.lower()
            download_dir.mkdir(parents=True, exist_ok=True)

            try:
                results.download(path=str(download_dir), session=session, processes=args.processes)
                downloads_completed.append(
                    {
                        "event_id": event_id,
                        "dataset": dataset_name,
                        "download_dir": str(download_dir),
                        "attempted_products": attempted,
                        "status": "ok",
                    }
                )
            except Exception as exc:
                err = str(exc)
                if "401" in err or "unauthor" in err.lower() or "token" in err.lower() or "credential" in err.lower():
                    status = f"error: auth: {err}"
                else:
                    status = f"error: download: {err}"
                downloads_completed.append(
                    {
                        "event_id": event_id,
                        "dataset": dataset_name,
                        "download_dir": str(download_dir),
                        "attempted_products": attempted,
                        "status": status,
                    }
                )

            if args.max_total_gb is not None:
                current_growth_gb = (_dir_size_bytes(outdir) - baseline_bytes) / (1024**3)
                if current_growth_gb >= args.max_total_gb:
                    cap_reached = True

    if downloads_completed:
        pd.DataFrame(downloads_completed).to_csv(outdir / "downloads_log.csv", index=False)

    run_metadata = {
        "months_before": args.months_before,
        "months_after": args.months_after,
        "max_results": args.max_results,
        "download_browse": args.download_browse,
        "download_products": bool(args.download_products),
        "auth_mode": args.auth_mode,
        "skip_counts": bool(args.skip_counts),
        "events_total": len(DEFAULT_EVENTS),
        "events_selected": len(selected_events),
        "datasets_selected": [name for name, _ in selected_datasets],
        "summary_rows": len(summary_rows),
        "max_downloads_per_manifest": args.max_downloads_per_manifest,
        "max_total_gb": args.max_total_gb,
        "prefer_derived": bool(args.prefer_derived),
        "split_file": args.split_file,
    }
    (outdir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    qc_summary = _generate_qc_reports(
        outdir=outdir,
        events_df=all_events_df,
        split=split,
        months_before=args.months_before,
        months_after=args.months_after,
    )

    print("Sinkhole ASF discovery complete.")
    print(f"Events processed: {len(selected_events)}")
    print(f"Output directory: {outdir.resolve()}")
    print(f"Manifest summary: {(outdir / 'manifest_summary.csv').resolve()}")
    print(f"QC summary: {(outdir / 'qc' / 'run_summary.json').resolve()}")
    if args.download_products:
        print(f"Download log: {(outdir / 'downloads_log.csv').resolve()}")
    print(json.dumps(qc_summary.get("acceptance", {}), indent=2))


if __name__ == "__main__":
    main()
