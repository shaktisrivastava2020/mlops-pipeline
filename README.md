# MLOps Pipeline

> Catches a model that improves overall but degrades for a customer segment — and refuses to promote it.

A production MLOps system that automates **drift detection**, **retraining**, and **fairness-aware promotion** for the [Day 2 churn predictor](https://github.com/shaktisrivastava2020/churn-predictor). Every step is audited, versioned, and reproducible.

**Live API:** https://mlops-api-218990051802.asia-south1.run.app
**Live demo dashboard:** https://mlops-api-218990051802.asia-south1.run.app/dashboard
**Interactive docs:** https://mlops-api-218990051802.asia-south1.run.app/docs

---

## Why this exists

Most MLOps tutorials stop at "deploy a model to Cloud Run." Real production starts the day after — when fresh data shifts, the model decays silently, and someone notices three months later that revenue is bleeding.

This project ships the layer that keeps a deployed model honest: scheduled drift detection, automated retraining, and **a promotion guard that refuses to ship a model that's worse for any customer segment** — even if overall metrics improve.

Most freelancers ship one model. A senior engineer ships the system that keeps the model good.

---

## The moment that matters

Trigger an `engagement_decline` drift scenario from the [dashboard](https://mlops-api-218990051802.asia-south1.run.app/dashboard). Three minutes later:

| | Champion (v0) | Challenger | Δ |
|---|---|---|---|
| Overall F1 | 0.4889 | 0.6667 | **+0.18** ✅ |
| `tier_premium` F1 | 0.6667 | 0.5000 | **−0.17** ❌ |
| **Decision** | | **REJECTED** | |

The challenger was 18 points better overall — well above the 0.02 promotion gate. But premium customers got *worse* by 17 points. **The system refused to promote it.**

> *"Segment 'tier_premium' regresses by -0.1667 (> 0.05)"*

That's the audit log line. That's the production-grade behavior: aggregate F1 doesn't ship a model on its own. Slice metrics do.

---

## What it does
┌──────────────────────┐
                │   Cloud Scheduler    │  weekly Mon 09:00 IST
                └──────────┬───────────┘
                           │ OIDC
                           ▼
Per pipeline run:

1. Pull fresh customer + order data from Cloud SQL
2. **Drift check** — KS for continuous, PSI for binary, multi-bin PSI for discrete
3. If drift detected → **retrain** using Day 2's exact architecture and hyperparameters
4. **Evaluate** champion vs challenger on the same holdout, including per-tier and per-tenure slices
5. **Promote** only if F1 improves by ≥ 0.02 AND no segment regresses by > 5%
6. **Log everything** to GCS — drift report, audit entry, alert webhook

---

## Drift detection — three tests, one decision

| Feature type | Test | Why |
|---|---|---|
| Continuous | Kolmogorov-Smirnov | sensitive to shifts in any quantile, including the tails where churn shows up first |
| Discrete (≤ 10 unique values) | PSI on observed bins | KS produces false positives on tied data; PSI on bin proportions is the right test |
| Binary | PSI on 0/1 proportions | clean signal for one-hot indicators |

**Two trigger rules — either trips drift:**

- **Fraction**: > 50% of features drift → retrain
- **Severity**: any single feature with PSI ≥ 0.25 OR KS ≥ 0.25 → retrain

A new payment method appearing in production trips the severity rule on `unique_payment_methods` alone — no need to wait for half the dataset to shift.

---

## Promotion criteria

A challenger is promoted **only if all** these hold:

1. Overall F1 ≥ champion F1 + **0.02** (prevents noise wins from random init)
2. No segment regresses by more than **5%** in F1 (the fairness guard)
3. Both evaluated on the same real-data holdout at threshold = 0.40

The 0.02 gate is calibrated to the dataset size (36-row holdout) — anything smaller is statistical noise. The 5% segment guard is what blocked the demo above.

---

## Quick start

```bash
# 1. Trigger a pipeline run with no drift
curl -X POST https://mlops-api-218990051802.asia-south1.run.app/run-pipeline \
  -H "Content-Type: application/json" \
  -d '{"skip_retraining": false}'

# 2. Inject drift, watch the system catch it
curl -X POST https://mlops-api-218990051802.asia-south1.run.app/run-pipeline \
  -H "Content-Type: application/json" \
  -d '{"inject_drift_preset": "engagement_decline"}'

# 3. Predict using the currently-promoted model
curl -X POST https://mlops-api-218990051802.asia-south1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_days": 365, "total_orders": 8, "total_spend": 80000,
    "avg_order_value": 10000, "days_since_last_order": 15, "orders_per_month": 2.0,
    "unique_products": 6, "unique_payment_methods": 3, "weekday_order_ratio": 0.7,
    "tier_standard": 0, "tier_premium": 1, "tier_gold": 0
  }'

# 4. See the audit trail
curl https://mlops-api-218990051802.asia-south1.run.app/audit/recent?limit=5
```

Or open the [dashboard](https://mlops-api-218990051802.asia-south1.run.app/dashboard) and click "Run pipeline" with a drift preset.

---

## Architecture

**Stack:** Python 3.11 · FastAPI 0.122 · PyTorch 2.5.1 (CPU) · scikit-learn 1.5.2 · SQLAlchemy 2.0 · Cloud SQL Python Connector · Multi-stage Docker · Cloud Run · GCS · Cloud Scheduler · Secret Manager

**Why each choice:**

- **PyTorch CPU-only** — Cloud Run has no GPU, GPU wheel pulls 2GB of nvidia deps we'd never use. CPU wheel is 250MB.
- **GCS as audit store** — append-only by `batch_id`, no extra database, every run is forever auditable.
- **Secret Manager for DB password** — never appears in image, env-var listings, deploy command, or shell history.
- **`min-instances=0`** — idle cost is $0. Cold-start ~5-10s, fine for non-latency-critical workloads.
- **OIDC between Scheduler and Cloud Run** — `/run-pipeline` is not exposed to random internet traffic.
- **Mirrored architecture from Day 2** (model.py, features.py, labeling.py) — verbatim copies, so champion-challenger comparisons are byte-equivalent.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/dashboard` | Visual demo dashboard with live trigger |
| `POST` | `/run-pipeline` | Trigger drift → retrain → evaluate → promote |
| `POST` | `/predict` | Predict using currently-promoted model |
| `GET` | `/models` | List all registered model versions |
| `GET` | `/audit/recent` | Last N pipeline-run audit entries |
| `GET` | `/model/info` | Current production version + metadata |
| `POST` | `/admin/reload-model` | Force re-read of production pointer |
| `GET` | `/health` | Liveness check |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Engineering practices

- **55 tests, all passing** — drift dispatcher, simulator presets, retrain reproducibility, evaluation slice math, promotion logic, API contracts
- **Reproducibility verified** — Day 3's retrain on real data matches Day 2's metrics exactly (best_epoch=8, val_loss=1.05, F1=0.50, pos_weight=2.75)
- **Versioned, immutable artifacts** — `models/v{N}/...` is write-once; only `production/current.json` is mutable
- **`pip-audit` clean** — 0 known CVEs across 80+ pinned dependencies
- **No PII anywhere** — schema documented separately, simulator generates only feature-relevant columns
- **Multi-stage Docker** — gcc + dev tools live in stage 1, runtime image is slim
- **Pre-commit discipline** — every commit reviewed for `.env`, model weights, secrets

---

## Cost

| Resource | Idle | Active |
|---|---|---|
| Cloud Run (mlops-api, min-instances=0) | $0 | $0.00002/req + CPU-secs |
| Cloud SQL (`db-f1-micro`) | $0 stopped | $11/mo running |
| GCS storage | < $0.01/mo | < $0.10/mo (full year of audits) |
| Cloud Scheduler | $0.10/mo | flat |
| Secret Manager | $0.06/mo per secret | flat |

**At rest, the entire system costs ~$0.20/month.** Stop Cloud SQL between sessions and the bill stays in pennies.

---

## Honest limitations

- **Small dataset.** 176 training rows, 36-row holdout. Confidence intervals on F1 are wide. Production-scale would loosen the 0.02 promotion gate proportional to sample size.
- **Synthetic drift simulator.** The simulator approximates Day 2's empirical distribution but doesn't perfectly reproduce it — there's irreducible baseline noise. Real production data would need threshold recalibration against actual false-positive rates.
- **No A/B traffic split.** Champion is replaced atomically. A real production system would shadow-deploy challengers to a small traffic slice before full promotion.
- **No concept-drift detection.** This system watches input distributions, not label distributions. A model could keep predicting confidently while the meaning of "churn" shifts in the business.
- **Single-region.** No multi-region failover.
- **Persist performance.** Simulator's batch-insert to Cloud SQL takes ~2 min for 400 rows; documented TODO, doesn't block any user flow.

---

## Repo structure
---

## Built by

**Shakti Srivastava**

- An AI optimist and a researcher
- GitHub: [@shaktisrivastava2020](https://github.com/shaktisrivastava2020)
- Open to freelance AI/ML projects — RAG, NL2SQL, NLP, Deep Learning, GenAI, ConvAI, Speech AI, Vision AI, Agentic, MCP, MLOps products on GCP & AWS
