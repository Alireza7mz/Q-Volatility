

"""
Tick-data preprocessing pipeline for medium-frequency quant project.

Stage 1: Load raw L1 quote ticks, compute 15 tick-level microstructure features.
Stage 2A: Build 1B Dollar volume buckets.
Stage 2B: Build 15-minute time bars.
Stage 3: Write monthly parquet files.

All configuration is imported from config.py. No hardcoded paths.
"""

from __future__ import annotations

import time
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from config import (
    RAW_DATA_DIR,
    MICRO_TICKS_DIR,
    MATRICES_DIR,
    ASSETS,
    VOLUME_BAR_SIZE,
    TIMEZONE,
    get_raw_csv_files,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Map raw CSV column names to canonical names
COLUMN_MAP: dict[str, str] = {
    "timestamp": "timestamp",
    "bid_price": "bid_price",
    "bid_size": "bid_size",
    "ask_price": "ask_price",
    "ask_size": "ask_size",
}

# Required raw columns (after renaming)
REQUIRED_RAW_COLUMNS: list[str] = [
    "timestamp",
    "bid_price",
    "bid_size",
    "ask_price",
    "ask_size",
]

# Canonical tick feature columns (15 total, in order)
TICK_FEATURE_COLUMNS: list[str] = [
    "timestamp",
    "time_delta",
    "bid_price",
    "bid_size",
    "ask_price",
    "ask_size",
    "spread",
    "raw_ofi",
    "obi",
    "mid_price",
    "micro_price",
    "micro_drift",
    "tick_return",
    "notional_volume",
    "tick_direction",
]

# Bar schema: exactly 28 columns in this order
BAR_SCHEMA: list[str] = [
    "bar_id",
    "start_time",
    "end_time",
    "session_date",
    "duration_seconds",
    "tick_count",
    "buy_ticks",
    "sell_ticks",
    "open",
    "high",
    "low",
    "close",
    "avg_spread",
    "min_spread",
    "max_spread",
    "std_spread",
    "avg_micro_price",
    "sum_raw_ofi",
    "abs_ofi",
    "avg_obi",
    "buy_volume",
    "sell_volume",
    "volume_delta",
    "total_notional_volume",
    "vwap",
    "realized_volatility",
    "path_length",
    "is_session_close",
]

# Per-tick features (used for aggregation in bar construction)
PER_TICK_FEATURES: list[str] = [
    "raw_ofi",
    "obi",
    "spread",
    "micro_price",
    "tick_return",
    "tick_direction",
]


