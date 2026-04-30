"""
Churn labeling — verbatim mirror of Day 2's labeling.py.

DO NOT edit independently. Any change must be reflected in both repos
or Day 3-retrained labels diverge from Day 2's training labels.

Rule: customer is churned if 2+ of these signals fire:
  1. RECENCY    — no engagement order in last 60 days
  2. FREQUENCY  — engagement orders in last 60d < 50% of prior 60d (prior >= 1)
  3. MONETARY   — spend in last 60d < 50% of prior 60d (prior > 0)
  4. NEGATIVE   — > 40% of orders are Cancelled or Returned
"""
from __future__ import annotations
import pandas as pd

WINDOW_DAYS = 60
DROP_THRESHOLD = 0.5
NEGATIVE_RATIO_THRESHOLD = 0.4
SIGNALS_REQUIRED = 2
ENGAGEMENT_STATUSES = ("Delivered", "Shipped", "Processing")
NEGATIVE_STATUSES = ("Cancelled", "Returned")


def compute_signals(orders_df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    if orders_df.empty:
        return pd.DataFrame()
    df = orders_df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    last_window_start = reference_date - pd.Timedelta(days=WINDOW_DAYS)
    prior_window_start = reference_date - pd.Timedelta(days=2 * WINDOW_DAYS)
    df["in_last"] = (df["order_date"] >= last_window_start) & (df["order_date"] < reference_date)
    df["in_prior"] = (df["order_date"] >= prior_window_start) & (df["order_date"] < last_window_start)
    eng = df[df["order_status"].isin(ENGAGEMENT_STATUSES)]

    agg = pd.DataFrame({"customer_id": df["customer_id"].unique()}).set_index("customer_id")

    last_order = eng.groupby("customer_id")["order_date"].max()
    agg["days_since_last_order"] = (reference_date - last_order).dt.days
    agg["signal_recency"] = (agg["days_since_last_order"] > WINDOW_DAYS).fillna(True).astype(int)

    last_freq = eng[eng["in_last"]].groupby("customer_id").size()
    prior_freq = eng[eng["in_prior"]].groupby("customer_id").size()
    agg["orders_last_60d"] = last_freq.reindex(agg.index, fill_value=0)
    agg["orders_prior_60d"] = prior_freq.reindex(agg.index, fill_value=0)
    agg["signal_frequency"] = (
        (agg["orders_prior_60d"] >= 1)
        & (agg["orders_last_60d"] < DROP_THRESHOLD * agg["orders_prior_60d"])
    ).astype(int)

    last_mon = eng[eng["in_last"]].groupby("customer_id")["order_amount"].sum()
    prior_mon = eng[eng["in_prior"]].groupby("customer_id")["order_amount"].sum()
    agg["spend_last_60d"] = last_mon.reindex(agg.index, fill_value=0).astype(float)
    agg["spend_prior_60d"] = prior_mon.reindex(agg.index, fill_value=0).astype(float)
    agg["signal_monetary"] = (
        (agg["spend_prior_60d"] > 0)
        & (agg["spend_last_60d"] < DROP_THRESHOLD * agg["spend_prior_60d"])
    ).astype(int)

    total_orders = df.groupby("customer_id").size()
    negative_orders = df[df["order_status"].isin(NEGATIVE_STATUSES)].groupby("customer_id").size()
    agg["total_orders"] = total_orders.reindex(agg.index, fill_value=0)
    agg["negative_orders"] = negative_orders.reindex(agg.index, fill_value=0)
    agg["negative_ratio"] = (agg["negative_orders"] / agg["total_orders"].replace(0, 1)).fillna(0)
    agg["signal_negative"] = (agg["negative_ratio"] > NEGATIVE_RATIO_THRESHOLD).astype(int)

    signal_cols = ["signal_recency", "signal_frequency", "signal_monetary", "signal_negative"]
    agg["signals_fired"] = agg[signal_cols].sum(axis=1)
    agg["churned"] = (agg["signals_fired"] >= SIGNALS_REQUIRED).astype(int)
    return agg.reset_index()
