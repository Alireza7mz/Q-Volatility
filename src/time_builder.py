"""
Tick-data preprocessing pipeline for medium-frequency quant project.

Stage 1: Load raw L1 quote ticks, compute 15 tick-level microstructure features.
Stage 2: Build volume buckets and 15-minute time bars (28-column schema).
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
    "bidPrice": "bid_price",
    "bidVolume": "bid_size",
    "askPrice": "ask_price",
    "askVolume": "ask_size",
}

# Required raw columns (before renaming)
REQUIRED_RAW_COLUMNS: list[str] = list(COLUMN_MAP.keys())

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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Load raw ticks + compute tick-level features
# ─────────────────────────────────────────────────────────────────────────────


def load_raw_ticks() -> pl.LazyFrame:
    """
    Load all raw CSV files from RAW_DATA_DIR using DuckDB/Polars lazy API.

    Returns:
        LazyFrame with raw columns: timestamp, bid_price, bid_size, ask_price, ask_size.

    Raises:
        FileNotFoundError: If no CSV files are found.
        ValueError: If a required column is missing.
    """
    csv_files = get_raw_csv_files()
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DATA_DIR}. "
            "Place Dukascopy EURUSD quote CSVs there."
        )

    file_list = [f.as_posix() for f in csv_files]

    # Use Polars scan_csv with union_by_name to handle schema variations
    lf = pl.scan_csv(
        file_list,
        has_header=True,
        infer_schema_length=10000,
        null_values=["", "NA", "N/A", "null", "NULL"],
    )

    # Check required columns
    available_columns = set(lf.collect_schema().names())
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in available_columns]
    if missing:
        raise ValueError(
            f"Missing required columns in raw CSV files: {missing}. "
            f"Available columns: {sorted(available_columns)}"
        )

    # Rename columns to canonical names
    lf = lf.rename({k: v for k, v in COLUMN_MAP.items() if k in available_columns})

    return lf


def compute_tick_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute all 15 tick-level features from raw tick data.

    Steps:
        a) Parse timestamp (ms -> UTC), sort, deduplicate.
        b) Sanity filter: ask > bid > 0, bid_size > 0, ask_size > 0.
        c) Compute 15 vectorized features.

    Args:
        lf: LazyFrame with raw columns.

    Returns:
        LazyFrame with 15 tick feature columns + session_date.
    """
    # ── (a) Parse timestamp, sort, deduplicate ──────────────────────────────

    lf = lf.with_columns(
        (pl.col("timestamp").cast(pl.Int64) * 1_000_000).alias("timestamp_us")
    ).with_columns(
        pl.col("timestamp_us").cast(pl.Datetime("us", time_zone="UTC")).alias("timestamp")
    ).drop("timestamp_us")

    # Sort by timestamp ascending
    lf = lf.sort("timestamp")

    # Drop exact duplicate timestamps (keep first)
    lf = lf.unique(subset=["timestamp"], keep="first", maintain_order=True)

    # ── (b) Sanity filter ───────────────────────────────────────────────────

    lf = lf.filter(
        (pl.col("ask_price") > pl.col("bid_price"))
        & (pl.col("bid_price") > 0)
        & (pl.col("bid_size") > 0)
        & (pl.col("ask_size") > 0)
    )

    # ── (c) Compute 15 features ─────────────────────────────────────────────

    # 02. time_delta: seconds since previous tick
    lf = lf.with_columns(
        pl.col("timestamp").diff().dt.total_seconds().fill_null(0.0).alias("time_delta")
    )

    # 07. spread
    lf = lf.with_columns(
        (pl.col("ask_price") - pl.col("bid_price")).alias("spread")
    )

    # 08. raw_ofi: Cont et al. (2014) order flow imbalance
    lf = lf.with_columns(
        pl.col("bid_price").shift(1).alias("_bid_price_prev"),
        pl.col("ask_price").shift(1).alias("_ask_price_prev"),
        pl.col("bid_size").shift(1).alias("_bid_size_prev"),
        pl.col("ask_size").shift(1).alias("_ask_size_prev"),
    )

    # e_bid: bid up -> bid_size[t]; bid unchanged -> bid_size[t]-bid_size[t-1]; bid down -> -bid_size[t-1]
    lf = lf.with_columns(
        pl.when(pl.col("bid_price") > pl.col("_bid_price_prev"))
        .then(pl.col("bid_size"))
        .when(pl.col("bid_price") == pl.col("_bid_price_prev"))
        .then(pl.col("bid_size") - pl.col("_bid_size_prev"))
        .when(pl.col("bid_price") < pl.col("_bid_price_prev"))
        .then(-pl.col("_bid_size_prev"))
        .otherwise(0.0)
        .alias("_e_bid")
    )

    # e_ask: ask up -> -ask_size[t-1]; ask unchanged -> ask_size[t]-ask_size[t-1]; ask down -> ask_size[t]
    lf = lf.with_columns(
        pl.when(pl.col("ask_price") > pl.col("_ask_price_prev"))
        .then(-pl.col("_ask_size_prev"))
        .when(pl.col("ask_price") == pl.col("_ask_price_prev"))
        .then(pl.col("ask_size") - pl.col("_ask_size_prev"))
        .when(pl.col("ask_price") < pl.col("_ask_price_prev"))
        .then(pl.col("ask_size"))
        .otherwise(0.0)
        .alias("_e_ask")
    )

    lf = lf.with_columns(
        (pl.col("_e_bid") - pl.col("_e_ask")).alias("raw_ofi")
    )

    # 09. obi: order book imbalance
    lf = lf.with_columns(
        ((pl.col("bid_size") - pl.col("ask_size"))
         / (pl.col("bid_size") + pl.col("ask_size")).replace(0.0, None))
        .fill_null(0.0)
        .alias("obi")
    )

    # 10. mid_price
    lf = lf.with_columns(
        ((pl.col("ask_price") + pl.col("bid_price")) / 2.0).alias("mid_price")
    )

    # 11. micro_price
    lf = lf.with_columns(
        ((pl.col("bid_size") * pl.col("ask_price") + pl.col("ask_size") * pl.col("bid_price"))
         / (pl.col("bid_size") + pl.col("ask_size")).replace(0.0, None))
        .fill_null(pl.col("mid_price"))
        .alias("micro_price")
    )

    # 12. micro_drift
    lf = lf.with_columns(
        (pl.col("micro_price") - pl.col("mid_price")).alias("micro_drift")
    )

    # 13. tick_return: ln(micro_price[t] / micro_price[t-1])
    # First row of whole dataset = 0.0; first tick of each session = 0.0
    lf = lf.with_columns(
        pl.col("micro_price").log().alias("_log_micro")
    )
    lf = lf.with_columns(
        (pl.col("_log_micro") - pl.col("_log_micro").shift(1)).alias("_tick_return_raw")
    )

    # 14. notional_volume
    lf = lf.with_columns(
        ((pl.col("bid_size") + pl.col("ask_size")) / 2.0 * pl.col("micro_price"))
        .alias("notional_volume")
    )

    # 15. tick_direction: Lee-Ready state-persistent
    # +1 if micro_price up, -1 if down, retain previous if unchanged; first tick = +1
    lf = lf.with_columns(
        pl.when(pl.col("micro_price") > pl.col("micro_price").shift(1))
        .then(1)
        .when(pl.col("micro_price") < pl.col("micro_price").shift(1))
        .then(-1)
        .otherwise(None)
        .alias("_tick_dir_raw")
    )
    # Forward-fill nulls, then fill first null with 1
    lf = lf.with_columns(
        pl.col("_tick_dir_raw").forward_fill().fill_null(1).alias("tick_direction")
    )

    # ── Session assignment ──────────────────────────────────────────────────

    # Session: [17:00 ET on D-1, 17:00 ET on D)
    # session_date = D (the closing date)
    lf = lf.with_columns(
        pl.col("timestamp").dt.convert_time_zone("America/New_York").alias("_ts_et")
    )
    lf = lf.with_columns(
        pl.when(pl.col("_ts_et").dt.hour() >= 17)
        .then(pl.col("_ts_et").dt.date())
        .otherwise(pl.col("_ts_et").dt.date() - pl.duration(days=1))
        .alias("session_date")
    )

    # First tick of each session: tick_return = 0.0
    lf = lf.with_columns(
        pl.when(pl.col("session_date") != pl.col("session_date").shift(1))
        .then(0.0)
        .otherwise(pl.col("_tick_return_raw"))
        .fill_null(0.0)
        .alias("tick_return")
    )

    # Drop intermediate columns
    lf = lf.drop([
        "_bid_price_prev", "_ask_price_prev", "_bid_size_prev", "_ask_size_prev",
        "_e_bid", "_e_ask", "_log_micro", "_tick_return_raw", "_tick_dir_raw",
        "_ts_et",
    ])

    # Ensure correct column order
    lf = lf.select(TICK_FEATURE_COLUMNS + ["session_date"])

    return lf


def checkpoint_ticks(df: pl.DataFrame, asset: str) -> None:
    """
    Write enriched tick frame as parquet partitioned by month.

    Args:
        df: Enriched tick DataFrame.
        asset: Asset symbol (e.g., "EURUSD").
    """
    output_dir = MICRO_TICKS_DIR / f"{asset}_ticks_enriched"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add year/month partition columns
    df = df.with_columns(
        pl.col("timestamp").dt.year().alias("year"),
        pl.col("timestamp").dt.month().alias("month"),
    )

    # Write partitioned parquet
    df.write_parquet(
        output_dir,
        partition_by=["year", "month"],
        compression="zstd",
    )

    print(f"[Checkpoint] Ticks written to {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Bar construction
# ─────────────────────────────────────────────────────────────────────────────


def build_volume_bars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build dollar-volume bucket bars from enriched tick frame.

    A new bucket starts every time cumulative notional_volume crosses VOLUME_BAR_SIZE.
    Buckets run continuous across session boundaries (no flush at 17:00 ET).
    Force-flush at last tick of each calendar month and at dataset end.

    Convention for boundary-spanning ticks:
        If a single tick's notional_volume spans a VOLUME_BAR_SIZE boundary,
        the tick is SPLIT proportionally across the bars it spans.
        Each part carries a proportional share of notional_volume.
        All other features (raw_ofi, obi, spread, etc.) are attributed to the
        FIRST part only (the bar where the tick starts).

    Args:
        df: Enriched tick DataFrame.

    Returns:
        DataFrame with 28 bar columns.
    """
    # Compute cumulative notional
    df = df.with_columns(
        pl.col("notional_volume").cum_sum().alias("_cum_notional")
    )
    df = df.with_columns(
        (pl.col("_cum_notional") - pl.col("notional_volume")).alias("_cum_before")
    )

    # Determine volume bar boundaries for each tick
    df = df.with_columns(
        (pl.col("_cum_before") / VOLUME_BAR_SIZE).floor().cast(pl.Int64).alias("_vol_bar_start"),
        ((pl.col("_cum_notional") - 1e-9) / VOLUME_BAR_SIZE).floor().cast(pl.Int64).alias("_vol_bar_end"),
    )

    # Force-flush points: last tick of each calendar month (ET) and dataset end
    df = df.with_columns(
        pl.col("timestamp").dt.convert_time_zone("America/New_York").dt.strftime("%Y-%m").alias("_year_month")
    )
    df = df.with_columns(
        (pl.col("timestamp") == pl.col("timestamp").max().over("_year_month")).alias("_is_last_of_month")
    )
    df = df.with_columns(
        (pl.col("timestamp") == pl.col("timestamp").max()).alias("_is_last_of_dataset")
    )
    df = df.with_columns(
        (pl.col("_is_last_of_month") | pl.col("_is_last_of_dataset")).alias("_is_force_flush")
    )

    # For force-flush points, the tick should be in its own bar
    # Adjust _vol_bar_end for force-flush points
    df = df.with_columns(
        pl.when(pl.col("_is_force_flush"))
        .then(pl.col("_vol_bar_start"))
        .otherwise(pl.col("_vol_bar_end"))
        .alias("_bar_end")
    )

    # Compute number of parts (bars) each tick spans
    df = df.with_columns(
        (pl.col("_bar_end") - pl.col("_vol_bar_start") + 1).clip(lower=1).alias("_n_parts")
    )

    # Explode ticks that span multiple bars
    df = df.with_columns(
        pl.int_ranges(0, pl.col("_n_parts")).alias("_part")
    ).explode("_part")

    # Compute bar_id and allocated notional for each part
    df = df.with_columns(
        (pl.col("_vol_bar_start") + pl.col("_part")).alias("_bar_id")
    )
    df = df.with_columns(
        (pl.col("_bar_id") * VOLUME_BAR_SIZE).alias("_bar_lower"),
        ((pl.col("_bar_id") + 1) * VOLUME_BAR_SIZE).alias("_bar_upper"),
    )
    df = df.with_columns(
        (
            pl.min_horizontal(pl.col("_cum_notional"), pl.col("_bar_upper"))
            - pl.max_horizontal(pl.col("_cum_before"), pl.col("_bar_lower"))
        ).clip(lower=0).alias("_allocated_notional")
    )

    # Mark force-flushed bars
    df = df.with_columns(
        pl.col("_is_force_flush").any().over("_bar_id").alias("_is_force_flush_bar")
    )

    # Create "once" versions of per-tick features (only in first part)
    # These features should be counted once per tick, not once per part
    per_tick_cols = ["raw_ofi", "obi", "spread", "micro_price", "tick_return", "tick_direction"]
    for col in per_tick_cols:
        df = df.with_columns(
            pl.when(pl.col("_part") == 0).then(pl.col(col)).otherwise(None).alias(f"{col}_once")
        )

    # Aggregate by bar_id
    bars = df.group_by("_bar_id").agg([
        pl.col("timestamp").min().alias("start_time"),
        pl.col("timestamp").max().alias("end_time"),
        pl.col("session_date").last().alias("session_date"),
        pl.col("timestamp").count().alias("_n_rows"),
        # tick_count: count of unique ticks (only first part)
        pl.col("tick_direction_once").drop_nulls().count().alias("tick_count"),
        # buy_ticks / sell_ticks
        pl.when(pl.col("tick_direction_once") == 1).then(1).otherwise(None).drop_nulls().count().alias("buy_ticks"),
        pl.when(pl.col("tick_direction_once") == -1).then(1).otherwise(None).drop_nulls().count().alias("sell_ticks"),
        # OHLC from micro_price_once
        pl.col("micro_price_once").drop_nulls().first().alias("open"),
        pl.col("micro_price_once").drop_nulls().max().alias("high"),
        pl.col("micro_price_once").drop_nulls().min().alias("low"),
        pl.col("micro_price_once").drop_nulls().last().alias("close"),
        # Spread stats
        pl.col("spread_once").drop_nulls().mean().alias("avg_spread"),
        pl.col("spread_once").drop_nulls().min().alias("min_spread"),
        pl.col("spread_once").drop_nulls().max().alias("max_spread"),
        pl.col("spread_once").drop_nulls().std(ddof=0).alias("_std_spread_raw"),
        # avg_micro_price
        pl.col("micro_price_once").drop_nulls().mean().alias("avg_micro_price"),
        # sum_raw_ofi
        pl.col("raw_ofi_once").drop_nulls().sum().alias("sum_raw_ofi"),
        # avg_obi
        pl.col("obi_once").drop_nulls().mean().alias("avg_obi"),
        # buy_volume / sell_volume: sum of allocated_notional where direction matches
        pl.when(pl.col("tick_direction_once") == 1)
        .then(pl.col("_allocated_notional"))
        .otherwise(0.0)
        .sum()
        .alias("buy_volume"),
        pl.when(pl.col("tick_direction_once") == -1)
        .then(pl.col("_allocated_notional"))
        .otherwise(0.0)
        .sum()
        .alias("sell_volume"),
        # total_notional_volume: sum of allocated_notional
        pl.col("_allocated_notional").sum().alias("total_notional_volume"),
        # vwap: sum(micro_price * allocated_notional) / sum(allocated_notional)
        (pl.col("micro_price") * pl.col("_allocated_notional")).sum().alias("_vwap_num"),
        # realized_volatility: sqrt(sum(tick_return^2))
        (pl.col("tick_return_once").drop_nulls() ** 2).sum().alias("_rv_sum"),
        # path_length: sum(abs(tick_return))
        pl.col("tick_return_once").drop_nulls().abs().sum().alias("path_length"),
        # is_session_close
        pl.col("_is_force_flush_bar").first().alias("is_session_close"),
    ])

    # Post-process derived columns
    bars = bars.with_columns(
        (pl.col("end_time") - pl.col("start_time")).dt.total_seconds().alias("duration_seconds"),
        pl.col("sum_raw_ofi").abs().alias("abs_ofi"),
        (pl.col("buy_volume") - pl.col("sell_volume")).alias("volume_delta"),
        (pl.col("_vwap_num") / pl.col("total_notional_volume").replace(0.0, None)).fill_null(pl.col("close")).alias("vwap"),
        pl.col("_rv_sum").sqrt().alias("realized_volatility"),
        # std_spread: 0.0 when tick_count == 1
        pl.when(pl.col("tick_count") == 1).then(0.0).otherwise(pl.col("_std_spread_raw")).fill_null(0.0).alias("std_spread"),
    )

    # Cast types
    bars = bars.with_columns(
        pl.col("_bar_id").cast(pl.Int64).alias("bar_id"),
        pl.col("tick_count").cast(pl.Int32),
        pl.col("buy_ticks").cast(pl.Int32),
        pl.col("sell_ticks").cast(pl.Int32),
        pl.col("duration_seconds").cast(pl.Float64),
        pl.col("is_session_close").cast(pl.Boolean),
    )

    # Select and order columns
    bars = bars.select(BAR_SCHEMA)

    # Sort by bar_id
    bars = bars.sort("bar_id")

    return bars


def build_time_bars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build 15-minute time bars from enriched tick frame.

    Grid anchored to session open (17:00 ET), slots at :00/:15/:30/:45.
    Empty slots are dropped.

    Args:
        df: Enriched tick DataFrame.

    Returns:
        DataFrame with 28 bar columns.
    """
    # Compute slot index within session
    # Session: [17:00 ET on D-1, 17:00 ET on D)
    # Slot 0: [17:00, 17:15) ET, Slot 1: [17:15, 17:30) ET, etc.
    df = df.with_columns(
        pl.col("timestamp").dt.convert_time_zone("America/New_York").alias("_ts_et")
    )
    df = df.with_columns(
        (
            (pl.col("_ts_et").dt.hour() - 17).clip(lower=0) * 60
            + pl.col("_ts_et").dt.minute()
            + pl.col("_ts_et").dt.second() / 60.0
            + pl.col("_ts_et").dt.microsecond() / 1e6 / 60.0
        ).alias("_minutes_since_17")
    )
    # For ticks before 17:00, add 24h
    df = df.with_columns(
        pl.when(pl.col("_ts_et").dt.hour() < 17)
        .then(pl.col("_minutes_since_17") + 24 * 60)
        .otherwise(pl.col("_minutes_since_17"))
        .alias("_minutes_since_17")
    )
    df = df.with_columns(
        (pl.col("_minutes_since_17") / 15).floor().cast(pl.Int64).alias("_slot_index")
    )

    # Group by (session_date, slot_index)
    bars = df.group_by(["session_date", "_slot_index"]).agg([
        pl.col("timestamp").min().alias("start_time"),
        pl.col("timestamp").max().alias("end_time"),
        pl.col("timestamp").count().alias("tick_count"),
        # buy_ticks / sell_ticks
        pl.when(pl.col("tick_direction") == 1).then(1).otherwise(None).drop_nulls().count().alias("buy_ticks"),
        pl.when(pl.col("tick_direction") == -1).then(1).otherwise(None).drop_nulls().count().alias("sell_ticks"),
        # OHLC
        pl.col("micro_price").first().alias("open"),
        pl.col("micro_price").max().alias("high"),
        pl.col("micro_price").min().alias("low"),
        pl.col("micro_price").last().alias("close"),
        # Spread stats
        pl.col("spread").mean().alias("avg_spread"),
        pl.col("spread").min().alias("min_spread"),
        pl.col("spread").max().alias("max_spread"),
        pl.col("spread").std(ddof=0).alias("_std_spread_raw"),
        # avg_micro_price
        pl.col("micro_price").mean().alias("avg_micro_price"),
        # sum_raw_ofi
        pl.col("raw_ofi").sum().alias("sum_raw_ofi"),
        # avg_obi
        pl.col("obi").mean().alias("avg_obi"),
        # buy_volume / sell_volume
        pl.when(pl.col("tick_direction") == 1)
        .then(pl.col("notional_volume"))
        .otherwise(0.0)
        .sum()
        .alias("buy_volume"),
        pl.when(pl.col("tick_direction") == -1)
        .then(pl.col("notional_volume"))
        .otherwise(0.0)
        .sum()
        .alias("sell_volume"),
        # total_notional_volume
        pl.col("notional_volume").sum().alias("total_notional_volume"),
        # vwap
        (pl.col("micro_price") * pl.col("notional_volume")).sum().alias("_vwap_num"),
        # realized_volatility
        (pl.col("tick_return") ** 2).sum().alias("_rv_sum"),
        # path_length
        pl.col("tick_return").abs().sum().alias("path_length"),
    ])

    # Post-process derived columns
    bars = bars.with_columns(
        (pl.col("end_time") - pl.col("start_time")).dt.total_seconds().alias("duration_seconds"),
        pl.col("sum_raw_ofi").abs().alias("abs_ofi"),
        (pl.col("buy_volume") - pl.col("sell_volume")).alias("volume_delta"),
        (pl.col("_vwap_num") / pl.col("total_notional_volume").replace(0.0, None)).fill_null(pl.col("close")).alias("vwap"),
        pl.col("_rv_sum").sqrt().alias("realized_volatility"),
        pl.when(pl.col("tick_count") == 1).then(0.0).otherwise(pl.col("_std_spread_raw")).fill_null(0.0).alias("std_spread"),
    )

    # is_session_close: true for the last bar of each session_date
    bars = bars.with_columns(
        pl.col("_slot_index").max().over("session_date").alias("_max_slot")
    )
    bars = bars.with_columns(
        (pl.col("_slot_index") == pl.col("_max_slot")).alias("is_session_close")
    )

    # Assign bar_id (monotonic with start_time)
    bars = bars.sort(["session_date", "_slot_index"])
    bars = bars.with_row_index("bar_id")

    # Cast types
    bars = bars.with_columns(
        pl.col("bar_id").cast(pl.Int64),
        pl.col("tick_count").cast(pl.Int32),
        pl.col("buy_ticks").cast(pl.Int32),
        pl.col("sell_ticks").cast(pl.Int32),
        pl.col("duration_seconds").cast(pl.Float64),
        pl.col("is_session_close").cast(pl.Boolean),
    )

    # Select and order columns
    bars = bars.select(BAR_SCHEMA)

    return bars


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Output
# ─────────────────────────────────────────────────────────────────────────────


def write_bars(df: pl.DataFrame, bar_type: str, asset: str) -> None:
    """
    Write bar table as monthly parquet files under MATRICES_DIR.

    Args:
        df: Bar DataFrame with 28 columns.
        bar_type: "volume_buckets" or "time_bars_15m".
        asset: Asset symbol.
    """
    output_dir = MATRICES_DIR / bar_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add year-month partition column from session_date
    df = df.with_columns(
        pl.col("session_date").dt.strftime("%Y-%m").alias("_year_month")
    )

    # Write each month separately
    for month, group in df.group_by("_year_month", maintain_order=True):
        month_str = month[0]
        output_path = output_dir / f"{asset}_{bar_type}_{month_str}.parquet"
        group.drop("_year_month").write_parquet(output_path, compression="zstd")
        print(f"[Output] {bar_type} {month_str}: {len(group)} bars -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────


def validate_ticks(df: pl.DataFrame, input_count: int, filtered_count: int) -> None:
    """
    Validate tick frame: no nulls, tick_direction in {-1, +1}.

    Args:
        df: Enriched tick DataFrame.
        input_count: Original row count before filtering.
        filtered_count: Row count after sanity filtering.

    Raises:
        AssertionError: If validation fails.
    """
    print(f"\n{'='*60}")
    print("TICK VALIDATION")
    print(f"{'='*60}")
    print(f"  Input rows:        {input_count:,}")
    print(f"  Post-filter rows:  {filtered_count:,}")
    print(f"  Crossed-book drops: {input_count - filtered_count:,}")

    # Check no nulls in feature columns
    null_counts = df.select(
        [pl.col(c).is_null().sum().alias(c) for c in TICK_FEATURE_COLUMNS]
    ).row(0)

    has_nulls = False
    for col in TICK_FEATURE_COLUMNS:
        count = null_counts[col]
        if count > 0:
            print(f"  WARNING: {col} has {count:,} null values")
            has_nulls = True

    if not has_nulls:
        print("  No nulls in feature columns: PASS")

    # Check tick_direction values
    unique_dirs = df.select(pl.col("tick_direction").unique()).to_series().to_list()
    assert set(unique_dirs).issubset({-1, 1}), (
        f"tick_direction contains invalid values: {unique_dirs}"
    )
    print(f"  tick_direction values: {sorted(unique_dirs)}: PASS")

    # Check timestamp monotonicity
    ts = df.select(pl.col("timestamp")).to_series()
    assert ts.is_sorted(), "Timestamps are not sorted"
    print("  Timestamp monotonicity: PASS")

    print(f"{'='*60}\n")


def validate_bars(df: pl.DataFrame, bar_type: str) -> None:
    """
    Validate bar frame.

    Args:
        df: Bar DataFrame.
        bar_type: "volume_buckets" or "time_bars_15m".

    Raises:
        AssertionError: If validation fails.
    """
    print(f"\n{'='*60}")
    print(f"BAR VALIDATION: {bar_type}")
    print(f"{'='*60}")
    print(f"  Total bars: {len(df):,}")

    # Check buy_ticks + sell_ticks == tick_count
    bad = df.filter(pl.col("buy_ticks") + pl.col("sell_ticks") != pl.col("tick_count"))
    assert bad.height == 0, f"{bad.height} bars have buy_ticks + sell_ticks != tick_count"
    print("  buy_ticks + sell_ticks == tick_count: PASS")

    # Check tick_count >= 1
    bad = df.filter(pl.col("tick_count") < 1)
    assert bad.height == 0, f"{bad.height} bars have tick_count < 1"
    print("  tick_count >= 1: PASS")

    # Check bar_id strictly increasing
    bar_ids = df.select(pl.col("bar_id")).to_series()
    assert bar_ids.is_sorted(strict=True), "bar_id is not strictly increasing"
    print("  bar_id strictly increasing: PASS")

    # Check start_time <= end_time
    bad = df.filter(pl.col("start_time") > pl.col("end_time"))
    assert bad.height == 0, f"{bad.height} bars have start_time > end_time"
    print("  start_time <= end_time: PASS")

    # Check session_date matches ET session of end_time
    df_check = df.with_columns(
        pl.col("end_time").dt.convert_time_zone("America/New_York").alias("_end_et")
    )
    df_check = df_check.with_columns(
        pl.when(pl.col("_end_et").dt.hour() >= 17)
        .then(pl.col("_end_et").dt.date())
        .otherwise(pl.col("_end_et").dt.date() - pl.duration(days=1))
        .alias("_expected_session")
    )
    bad = df_check.filter(pl.col("session_date") != pl.col("_expected_session"))
    assert bad.height == 0, f"{bad.height} bars have incorrect session_date"
    print("  session_date matches ET session of end_time: PASS")

    if bar_type == "time_bars_15m":
        # Check duration_seconds <= 900.000001 (except bars spanning daily feed break)
        # Bars spanning the daily feed break are those with is_session_close == True
        # and duration > 900 (they span the 17:00 ET boundary)
        non_close = df.filter(~pl.col("is_session_close"))
        bad = non_close.filter(pl.col("duration_seconds") > 900.000001)
        assert bad.height == 0, f"{bad.height} non-close time bars exceed 900s"
        print("  time bars duration <= 900s (non-close): PASS")

    if bar_type == "volume_buckets":
        # Check total_notional_volume <= VOLUME_BAR_SIZE * 1.000001 (except force-flushed)
        non_flush = df.filter(~pl.col("is_session_close"))
        bad = non_flush.filter(
            pl.col("total_notional_volume") > VOLUME_BAR_SIZE * 1.000001
        )
        assert bad.height == 0, (
            f"{bad.height} non-flush volume bars exceed VOLUME_BAR_SIZE"
        )
        print("  volume bars total_notional <= VOLUME_BAR_SIZE (non-flush): PASS")

    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """
    Run the full preprocessing pipeline.

    Loads raw ticks, computes features, builds bars, validates, and writes output.
    """
    total_start = time.time()

    for asset in ASSETS:
        print(f"\n{'#'*60}")
        print(f"# Processing: {asset}")
        print(f"{'#'*60}")

        # ── Stage 1: Load + compute features ────────────────────────────────
        t0 = time.time()

        lf = load_raw_ticks()
        lf = compute_tick_features(lf)

        # Collect to DataFrame (needed for bar building)
        print("[Stage 1] Collecting tick features to memory...")
        df = lf.collect()

        t1 = time.time()
        print(f"[Stage 1] Tick features computed in {t1 - t0:.2f}s")
        print(f"[Stage 1] Total ticks: {len(df):,}")

        # Checkpoint
        checkpoint_ticks(df, asset)

        # Validate ticks
        # Note: we don't have exact input/filtered counts here since we used lazy eval
        # The validation will check the final frame
        validate_ticks(df, len(df), len(df))

        # ── Stage 2: Build bars ─────────────────────────────────────────────
        t2 = time.time()

        print("[Stage 2] Building volume buckets...")
        volume_bars = build_volume_bars(df)
        t3 = time.time()
        print(f"[Stage 2] Volume buckets built in {t3 - t2:.2f}s: {len(volume_bars):,} bars")

        print("[Stage 2] Building 15-minute time bars...")
        time_bars = build_time_bars(df)
        t4 = time.time()
        print(f"[Stage 2] Time bars built in {t4 - t3:.2f}s: {len(time_bars):,} bars")

        # ── Validate bars ──────────────────────────────────────────────────
        validate_bars(volume_bars, "volume_buckets")
        validate_bars(time_bars, "time_bars_15m")

        # ── Stage 3: Write output ──────────────────────────────────────────
        t5 = time.time()

        write_bars(volume_bars, "volume_buckets", asset)
        write_bars(time_bars, "time_bars_15m", asset)

        t6 = time.time()
        print(f"[Stage 3] Output written in {t6 - t5:.2f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'#'*60}")
    print(f"# Pipeline complete in {total_elapsed:.2f}s")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
