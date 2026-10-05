


# data_loader.py
import duckdb
from config import get_raw_csv_files, MICRO_TICKS_DIR, ASSETS


def load_and_cache_raw_ticks(symbol: str = "EURUSD") -> None:
    """
    Load raw CSV files using DuckDB's out-of-core streaming engine.
    Applies column mapping, timestamp conversion (epoch ms -> UTC),
    deduplication, sorting, and exports to compressed Parquet.
    """
    csv_files = get_raw_csv_files()
    if not csv_files:
        raise FileNotFoundError(f"[DataLoader] No CSV files found in the raw directory!")

    print(f"[DataLoader] Reading {len(csv_files)} raw CSV files for {symbol} with DuckDB...")

    # Convert paths to POSIX format to prevent Windows backslash escaping errors
    file_list = [f.as_posix() for f in csv_files]
    output_path = (MICRO_TICKS_DIR / f"{symbol}_raw_ticks.parquet").as_posix()

    # DuckDB Query: Stream CSVs -> Convert -> Deduplicate -> Sort -> Stream to Parquet
    query = f"""
        COPY (
            SELECT DISTINCT ON (timestamp)
                timestamp,
                bid,
                ask,
                bid_size,
                ask_size
            FROM (
                SELECT
                    epoch_ms(TRY_CAST(timestamp AS BIGINT)) AS timestamp,
                    TRY_CAST(bidPrice AS DOUBLE) AS bid,
                    TRY_CAST(askPrice AS DOUBLE) AS ask,
                    TRY_CAST(bidVolume AS DOUBLE) AS bid_size,
                    TRY_CAST(askVolume AS DOUBLE) AS ask_size
                FROM read_csv_auto({file_list}, header=True, union_by_name=True)
            )
            WHERE timestamp IS NOT NULL
            ORDER BY timestamp ASC
        )
        TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    con = duckdb.connect()
    con.execute(query)
    con.close()

    print(f"[DataLoader] L1 checkpoint successfully built at: {output_path}")


if __name__ == "__main__":
    for asset in ASSETS:
        load_and_cache_raw_ticks(asset)
        print(f"L1 checkpoint built for {asset}")





