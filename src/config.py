

from pathlib import Path

# ==========================================
# 1. Path Management
# ==========================================
# Project root path on your system
BASE_DIR = Path(__file__).resolve().parent

# Base data directory
DATA_DIR = Path("/Users/alireza/Documents/data")

# Input and output directories
RAW_DATA_DIR = DATA_DIR / "eur" / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "eur" / "processed"

# Refined data layers
MICRO_TICKS_DIR = PROCESSED_DATA_DIR / "01_micro_ticks"
MATRICES_DIR = PROCESSED_DATA_DIR / "02_matrices"

# Automatically create output folders if they do not exist
for folder in [MICRO_TICKS_DIR, MATRICES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Operational Parameters and Assets
# ==========================================
ASSETS = ["EURUSD"]  # Later, only add new names like "XAUUSD" to this list

# Bar and layer settings
VOLUME_BAR_SIZE = 1_000_000_000  # 1000 million units for EUR
TIMEZONE = "UTC"

# ==========================================
# 3. Path Helper Functions (Dynamic Helpers)
# ==========================================
def get_raw_csv_files():
    """Read all CSV files in the raw folder without specifying names"""
    return sorted(list(RAW_DATA_DIR.glob("*.csv")))







# -------------------------------------------------- Time bar config --------------------------------------------------


from dataclasses import dataclass
from typing import Optional



@dataclass(frozen=True)
class TimeBarConfig:
    """
    Configuration dataclass for constructing flexible time-aggregated bars
    from pre-computed tick-level feature dataframes (tick_features_df).
    """
    symbol: str
    target_tz: str = "America/New_York"       # Target timezone for session boundary alignment (handles DST automatically)
    cutoff_hour: int = 17                     # NY session cutoff hour (17:00 NY for Forex/Gold day transition)
    bar_size: str = "1D"                      # Bar resolution string ('1D', '4H', '1H', '15T', '1min')
    price_col: str = "micro_price"            # Reference price column for OHLC computation
    vol_col: str = "notional_volume"          # Reference volume column for aggregation and VWAP
    max_spread: Optional[float] = None        # Threshold for filtering anomalous spread ticks
    include_microstructure: bool = True       # Toggle for computing aggregated L1 microstructure metrics (OFI, OBI, etc.)

    def __post_init__(self):
        """Validates configuration parameters upon instantiation."""
        if not (0 <= self.cutoff_hour <= 23):
            raise ValueError(f"Invalid cutoff_hour: {self.cutoff_hour}. Must be within [0, 23].")
            
        valid_units = ("D", "H", "T", "min", "S")
        if not any(unit in self.bar_size for unit in valid_units):
            raise ValueError(
                f"Invalid bar_size: '{self.bar_size}'. Must use standard Pandas frequency aliases (e.g., '1D', '4H', '15T')."
            )




