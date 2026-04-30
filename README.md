# MLOps Pipeline

Production MLOps system that automates **drift detection**, **retraining**, **champion/challenger evaluation**, and **versioned promotion** for the [Day 2 churn predictor](https://github.com/shaktisrivastava2020/churn-predictor).

**Live API:** https://mlops-api-218990051802.asia-south1.run.app
**Docs:** https://mlops-api-218990051802.asia-south1.run.app/docs

---

## What it does

1. Runs on a weekly schedule (Cloud Scheduler) or on demand (`POST /run-pipeline`).
2. Pulls fresh customer + order data from Cloud SQL.
3. **Drift check**: compares current distribution to the frozen reference using KS test (continuous), PSI (binary, discrete). Two trigger rules — fraction-based + severity-based.
4. **Retrains** the same architecture as Day 2 (32-16-1 ChurnNet, BatchNorm, weighted BCE) if drift is detected.
5. **Evaluates** champion vs challenger on a shared holdout, including **per-tier and per-tenure slice analysis**.
6. **Promotes** only if (a) F1 improves by ≥ 0.02 AND (b) no segment regresses by > 5%.
7. Persists every step (drift report, audit log, model artifacts) to GCS for audit trail.
8. Sends Slack alerts on drift / promotion / rejection / errors.

## Architecture
┌──────────────────┐
             │ Cloud Scheduler  │  (weekly Mon 09:00 IST)
             └────────┬─────────┘
                      │ OIDC POST
                      ▼
## Stack

- FastAPI 0.122 + Pydantic 2.10
- PyTorch 2.5.1 (CPU) + scikit-learn 1.5.2
- Cloud Run (multi-stage Docker, `min-instances=0`)
- Cloud SQL Postgres (shared with Day 1+2)
- GCS for model registry + audit logs
- Cloud Scheduler for weekly trigger
- 55 unit + integration tests, pip-audit clean

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/model/info` | Current production version + metadata |
| POST | `/predict` | Predict churn using promoted model |
| POST | `/run-pipeline` | Trigger one full pipeline run |
| GET | `/models` | List all registered model versions |
| GET | `/audit/recent` | Last N pipeline-run audit entries |
| POST | `/admin/reload-model` | Force re-read of production pointer |

## Drift detection

Three tests dispatched by feature type:

| Feature type | Test | Threshold |
|---|---|---|
| Continuous | Kolmogorov-Smirnov | p-value < 0.05 |
| Discrete (≤10 unique values) | PSI on observed bins | PSI > 0.20 |
| Binary | PSI on 0/1 proportions | PSI > 0.20 |

Two system-level decision rules:
- **Fraction**: > 50% of features drift → retrain (calibrated against synthetic baseline noise)
- **Severity**: any single feature with PSI ≥ 0.25 OR KS ≥ 0.25 → retrain

Either rule trips → drift detected. The audit log records which rule fired.

## Promotion criteria

A challenger is promoted **only if all** of these hold:

1. Overall F1 ≥ champion F1 + 0.02
2. No segment (per-tier, per-tenure) regresses by > 5% in F1
3. Both evaluated on the same real-data holdout with threshold = 0.40

This is the fairness guard: a model that improves overall but degrades for a customer segment is **rejected**, not silently promoted.

## Repo layout
## Costs (estimated)

- Cloud Run: ~$0/month (min-instances=0, scales to zero between requests)
- Cloud SQL: $11/month if always-on, ~$0 if stopped between sessions
- GCS storage: < $1/month for model artifacts + logs
- Cloud Scheduler: $0.10/month per job
- **Realistic total**: < $1/month for portfolio use, ~$15/month for production-style always-on

## Limitations & honest caveats

- **Synthetic baseline noise**: the data simulator approximates Day 2's empirical distribution but doesn't perfectly reproduce it. Drift thresholds are calibrated against this baseline noise (~30% feature-noise floor). On real production data the thresholds may need recalibration.
- **Small dataset**: 176 training samples, 36 test. Confidence intervals on F1 are wide. The promotion gate (0.02 F1 improvement) is conservative for the dataset size.
- **`weekday_order_ratio` false positive**: this feature is a low-cardinality ratio that occasionally trips KS even on baseline data. Documented; falls under the noise floor.
- **No A/B traffic split**: champion is replaced atomically. A real production system would shadow-deploy challengers.
- **Single-region**: not designed for multi-region failover.

## Setup (reproduce)

```bash
git clone https://github.com/shaktisrivastava2020/mlops-pipeline.git
cd mlops-pipeline
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
# Set DB_PASSWORD in .env (chmod 600)
python compute_reference_stats.py    # only if reference_stats.json regenerated
pytest tests/ -q                     # 55 tests
uvicorn main:app --reload            # local dev server
```

## What I built (1-line summary)

A scheduled, audit-logged, fairness-aware MLOps pipeline that catches a model regressing on a customer segment and refuses to promote it — even when overall metrics look better. Demonstrated end-to-end on production Cloud Run with weekly Cloud Scheduler trigger.
