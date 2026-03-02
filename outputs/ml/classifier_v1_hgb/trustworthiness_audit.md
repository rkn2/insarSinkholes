# Trustworthiness Audit

- Verdict: **high_leakage_risk**
- Findings count: **1**

## Findings
- `days_to_event` is inference-leaky for real-time deployment.

## Baseline Test Metrics
- Precision: 0.9811
- Recall: 0.9811
- F1: 0.9811

## Counterfactual (Leakage Features Removed)
- Removed: days_to_event, robust_accel_z, robust_vel_z
- Test F1: 0.4516
- Test Precision: 0.3431
- Test Recall: 0.6604

## Recommended Actions
- Remove days_to_event from deployment model features.
- Do not define labels using robust_vel_z/robust_accel_z if those are model inputs.
- Rebuild labels from external event windows only, then recompute model features independently.
- Re-run event-wise CV after leakage feature removal and compare holdout stability.
