# Repository Evaluation for Sinkhole Monitoring (2026-03-01)

## Scope

This review evaluated the repositories/tools you listed for practical integration into this project:

- `insarlab/MintPy`
- `dbekaert/StaMPS`
- `yumorishita/LiCSBAS2`
- `TUDelftGeodesy/DePSI`
- `isce-framework/dolphin`
- `alexisInSAR/EZ-InSAR`
- `CNES/S1Tiling`
- `osherr1996/SinkSAM`
- `sinkholenet/sinkholenet`
- `Hyradus/DeepLandforms`
- `LyuBaihang2024/InSAR-RiskLSTM`

Two names from your list could not be mapped to a clear, maintained public GitHub source during this pass:

- `IKCPNet`
- `Sinkhole-Extraction-Tool` (ArcGIS toolbox reference, but no clearly canonical public GitHub repo identified in this run)

## Validation Method

Validation was done on 2026-03-01 with:

1. GitHub repository metadata checks (stars, push dates, license, archive status).
2. Shallow clone + repository structure scan.
3. Packaging/dependency inspection.
4. Concrete execution attempts in this environment (install/CLI/tests).

## High-Level Result

`MintPy` is currently the best immediate path for this codebase.

It was the only major InSAR engine in this pass that installed and ran a CLI command successfully with minimal friction in the current environment.

## Practical Results

### 1) MintPy (`insarlab/MintPy`)

- Status: `PASS` (install + CLI run succeeded)
- Evidence:
  - `python3 -m pip install -q mintpy` succeeded.
  - `python3 -m mintpy --help` succeeded and exposed full command suite.
- Why it ranks highest:
  - Mature ecosystem and broad ingestion support.
  - Direct path to time-series products needed by your current pipeline.

### 2) dolphin (`isce-framework/dolphin`)

- Status: `PARTIAL` (install succeeded, runtime blocked)
- Evidence:
  - `python3 -m pip install -q dolphin` succeeded.
  - `python3 -m dolphin --help` failed first on missing `rich`, then on missing `osgeo` (GDAL runtime).
- Interpretation:
  - Strong candidate, but requires geospatial system dependencies (GDAL/OSGeo stack) before practical use.

### 3) LiCSBAS2 (`yumorishita/LiCSBAS2`)

- Status: `BLOCKED` (runtime dependency)
- Evidence:
  - Running a script entrypoint failed with `ModuleNotFoundError: osgeo`.
- Interpretation:
  - Good regional Sentinel-1 workflow, but not immediately runnable here without geospatial system setup.

### 4) S1Tiling (`CNES/S1Tiling`)

- Status: `BLOCKED` (packaging/dependency conflicts in this environment)
- Evidence:
  - `python3 -m pip install -q S1Tiling` failed with dependency resolution and legacy GDAL build issues.
- Interpretation:
  - Useful utility, but setup complexity is high in current environment.

### 5) DePSI (`TUDelftGeodesy/DePSI`)

- Status: `PARTIAL/BLOCKED`
- Evidence:
  - Install from source with `--user` succeeded.
  - Test execution required additional dependencies (`pytest`, `pytest-cov`, `xarray`) and failed during collection due missing `xarray`.
- Interpretation:
  - Promising modern PS implementation, but not plug-and-play here yet.

### 6) InSAR-RiskLSTM (`LyuBaihang2024/InSAR-RiskLSTM`)

- Status: `BLOCKED` (dependency pins)
- Evidence:
  - Requirements install failed due `opencv-python==4.7.0` pin mismatch.
  - Tests failed due missing `torch`.
- Interpretation:
  - Research codebase with strict pinning; requires controlled env refresh and dependency modernization.

### 7) SinkSAM (`osherr1996/SinkSAM`)

- Status: `RESEARCH-ONLY (not production-ready in this pass)`
- Evidence:
  - Repository is lightweight and paper-focused; no robust packaging/test harness detected in this pass.
- Interpretation:
  - Valuable for segmentation methodology ideas, but not yet an immediate production module.

### 8) sinkholenet (`sinkholenet/sinkholenet`)

- Status: `RESEARCH DATASET/FRAMEWORK REFERENCE`
- Evidence:
  - Very small repo footprint; appears as paper/data reference rather than production package.
- Interpretation:
  - Useful conceptually, limited direct operational utility for your InSAR pipeline.

### 9) DeepLandforms (`Hyradus/DeepLandforms`)

- Status: `RESEARCH-ORIENTED`
- Evidence:
  - Mentions sinkhole-like landforms (Mars validation context), limited production packaging signals.
- Interpretation:
  - Better as model inspiration than immediate operational component.

### 10) StaMPS (`dbekaert/StaMPS`) and EZ-InSAR (`alexisInSAR/EZ-InSAR`)

- Status: `NOT EXECUTED IN THIS PASS` (environment/tooling mismatch risk)
- Interpretation:
  - Both are important in practice, but they require heavier MATLAB/legacy processing chain assumptions than your current Python-first project setup.

## What Works Best for This Repository Right Now

Recommended adoption order:

1. `MintPy` as the core InSAR time-series engine.
2. `dolphin` as a next-stage enhancement once GDAL/OSGeo stack is standardized.
3. `S1Tiling` only if you commit to radar-optical co-registration workflows and can support OTB/GDAL tooling.
4. Sinkhole-specific ML repos (`SinkSAM`, `InSAR-RiskLSTM`) as later-stage R&D modules after you create a local labeled training set and isolated ML environment.

## Validation Confidence and Limits

What was validated concretely:

- Real install/run attempts for `MintPy`, `dolphin`, `S1Tiling`, `InSAR-RiskLSTM`, and partial for `DePSI`.
- Repository-level inspection for all resolved repos.

What was not fully validated:

- End-to-end PSI/SBAS processing on a full interferometric stack from these external frameworks.
- Model accuracy benchmarking across SinkSAM/SinkholeNet/DeepLandforms due dataset/training prerequisites.
- `IKCPNet` and `Sinkhole-Extraction-Tool` due unresolved canonical source mapping in this pass.

## Next Validation Steps (Actionable)

1. Stand up a dedicated conda environment for geospatial stack (`gdal`, `rasterio`, `pyproj`) and re-run `dolphin` + `LiCSBAS2` smoke tests.
2. Run one shared benchmark AOI through `MintPy` and your current upgraded pipeline, then compare:
   - first-alert date,
   - false-alarm rate per year,
   - lead time to known events.
3. Build a minimum training set (even 100-300 labeled tiles) before attempting SinkSAM/InSAR-RiskLSTM integration.
4. Add a reproducible benchmark script that logs metrics for each pipeline variant into one CSV for apples-to-apples comparison.
