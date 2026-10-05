


# Data processing and numerical computations
import numpy as np
import pandas as pd
import math
from collections import Counter



from data_loader import get_cleaned_data    #######   change this to the path of the data_loader.py file
daily_df = get_cleaned_data(timeframe="1D")

hourly_df = get_cleaned_data(timeframe="1h")




# Feature List

# 01-CALENDAR & SEASONALITY FEATURES
# 02-INTRADAY DYNAMICS - CLOSE LOCATION VALUE (CLV) 
# 03-Realized_Skew_ZScore , Realized_Kurt_ZScore
# 04-ROLLING DETRENDED FLUCTUATION ANALYSIS (DFA) SCALING EXPONENT
# 05-Range_CV_22D

# =================================================================================================== CALENDAR & SEASONALITY FEATURES



# 1. Day of Week (0 = Monday, 6 = Sunday)
daily_df['day_of_week'] = daily_df.index.dayofweek

# Cyclical Encoding for Day of Week (Preserves continuous cycle between Sun & Mon)
daily_df['sin_day_of_week'] = np.sin(2 * np.pi * daily_df['day_of_week'] / 7.0)
daily_df['cos_day_of_week'] = np.cos(2 * np.pi * daily_df['day_of_week'] / 7.0)

# 2. Day of Month (1 to 31) & Cyclical Encoding
daily_df['day_of_month'] = daily_df.index.day
daily_df['sin_day_of_month'] = np.sin(2 * np.pi * daily_df['day_of_month'] / 31.0)
daily_df['cos_day_of_month'] = np.cos(2 * np.pi * daily_df['day_of_month'] / 31.0)

# 3. Month of Year (1 to 12) & Cyclical Encoding
daily_df['month'] = daily_df.index.month
daily_df['sin_month'] = np.sin(2 * np.pi * daily_df['month'] / 12.0)
daily_df['cos_month'] = np.cos(2 * np.pi * daily_df['month'] / 12.0)

# 4. Financial Calendar Rebalancing / Expiration Flags (0 or 1)
daily_df['is_month_end'] = daily_df.index.is_month_end.astype(int)
daily_df['is_quarter_end'] = daily_df.index.is_quarter_end.astype(int)





# 1. Continuous Month Progress (0.0 to 1.0)
daily_df["f_month_progress"] = daily_df.index.day / daily_df.index.days_in_month

# 2. Continuous Quarter Progress (0.0 to 1.0) - Tz-Safe
idx_naive = (
    daily_df.index.tz_localize(None)
    if daily_df.index.tz is not None
    else daily_df.index
)

q_start = idx_naive.to_period("Q").start_time
q_end = idx_naive.to_period("Q").end_time

daily_df["f_quarter_progress"] = (idx_naive - q_start) / (q_end - q_start)


# =================================================================================================== INTRADAY DYNAMICS - CLOSE LOCATION VALUE (CLV) 


daily_range = daily_df["High"] - daily_df["Low"]

# Handle potential division by zero (flat candles where High == Low)
daily_df["f_clv"] = np.where(
    daily_range > 0,
    (daily_df["Close"] - daily_df["Low"]) / daily_range,
    0.5,  # Default neutral value if no movement occurs
)



# =================================================================================================== Range 


daily_df["Range"] = daily_df["High"] - daily_df["Low"] 

daily_df["Range_Lag1"] = daily_df["Range"].shift(1)

daily_df["Tomorrow_Range"] = daily_df["Range"].shift(-1)


# =================================================================================================== Average Daily Range



daily_df["ADR_5"] = daily_df["Range"].rolling(window=5).mean()
daily_df["ADR_20"] = daily_df["Range"].rolling(window=20).mean()

daily_df['ADR_66'] = daily_df['Range'].rolling(window=66).mean()


daily_df["Range_STD_5"] = daily_df["Range"].rolling(window=5).std()
daily_df["Range_STD_20"] = daily_df["Range"].rolling(window=20).std()



daily_df["Persistence"] =  daily_df["Range_Lag1"]  / daily_df["Range_STD_20"] 

daily_df['Relative_Range'] = daily_df['Range'] / daily_df['ADR_20']



# =================================================================================================== Return 



daily_df["Return"] = ( (daily_df["Close"] - daily_df["Open"]) / daily_df["Open"] )



# =================================================================================================== Body 


daily_df["Body"] = daily_df["Close"] - daily_df["Open"]

daily_df["Abs_Body"] = daily_df["Body"].abs() 

daily_df["Tomorrow_Abs_Body"] = daily_df["Abs_Body"].shift(-1)

daily_df['Body_Range_Ratio'] = daily_df['Abs_Body'] / daily_df['Range']


# =================================================================================================== Efficiency 


daily_df["Efficiency"] = daily_df["Abs_Body"] / daily_df["Range"]



# =================================================================================================== RS varianc


# Step 1: Calculate single-bar RS variance
u = np.log(daily_df['High'] / daily_df['Open'])
d = np.log(daily_df['Low']  / daily_df['Open'])
c = np.log(daily_df['Close']/ daily_df['Open'])


# Floor variance at 0 to prevent NaN in sqrt (Essential defensive line)
daily_df['RS_Var'] = np.maximum(u * (u - c) + d * (d - c), 0)




# Step 2: 20-day rolling mean + square root (exact ATR replacement)
daily_df['RS_Vol_1D']  = np.sqrt(daily_df['RS_Var'])
daily_df['RS_Vol_5D']  = np.sqrt(daily_df['RS_Var'].rolling(window=5).mean())
daily_df['RS_Vol_22D'] = np.sqrt(daily_df['RS_Var'].rolling(window=22).mean())



# Step 3: Natural Log Transformation (To achieve Gaussian distribution)
daily_df['Log_RS_Vol_1D']  = np.log(daily_df['RS_Vol_1D']  + 1e-8)
daily_df['Log_RS_Vol_5D']  = np.log(daily_df['RS_Vol_5D']  + 1e-8)
daily_df['Log_RS_Vol_22D'] = np.log(daily_df['RS_Vol_22D'] + 1e-8)




# Step 4: Rolling standardization (Z-Score with 90-day window)
window_z_RS = 90

# Z-Score for 1D (Instantaneous / Today's Volatility)
mean_1d = daily_df['Log_RS_Vol_1D'].rolling(window=window_z_RS).mean()
std_1d  = daily_df['Log_RS_Vol_1D'].rolling(window=window_z_RS).std()
daily_df['Log_RS_Vol_1D_ZScore'] = ((daily_df['Log_RS_Vol_1D'] - mean_1d) / std_1d).clip(-3.5, 3.5)


# Z-Score for 22D (Monthly Volatility Level)
mean_22d = daily_df['Log_RS_Vol_22D'].rolling(window=window_z_RS).mean()
std_22d  = daily_df['Log_RS_Vol_22D'].rolling(window=window_z_RS).std()
daily_df['Log_RS_Vol_22D_ZScore'] = ((daily_df['Log_RS_Vol_22D'] - mean_22d) / std_22d).clip(-3.5, 3.5)






# 1. Volatility Term Structure / Ratio (Weekly to Monthly)
# Note: Division in real space = subtraction in log space
daily_df['Vol_Term_Structure_5_22'] = daily_df['Log_RS_Vol_5D'] - daily_df['Log_RS_Vol_22D']


# 2. Volatility Divergence (Today vs Weekly)
daily_df['Vol_Divergence_1_5'] = daily_df['Log_RS_Vol_1D'] - daily_df['Log_RS_Vol_5D']


# 3. Lag-1 Range Autocorrelation (30-day rolling correlation of daily volatility)
daily_df['Log_RS_Autocorr1'] = (
    daily_df['Log_RS_Vol_1D']
    .rolling(window=30)
    .corr(daily_df['Log_RS_Vol_1D'].shift(1))
    .fillna(0)
)




# =================================================================================================== Kaufman Efficiency Ratio (KER)

# Time horizon (e.g., 22 days for monthly)
window_ker = 22

# 1. Numerator: net price change over the last 22 days (absolute)
net_change = (daily_df['Close'] - daily_df['Close'].shift(window_ker)).abs()

# 2. Denominator: sum of absolute daily changes (path length) over the last 22 days
abs_daily_change = (daily_df['Close'] - daily_df['Close'].shift(1)).abs()
path_length = abs_daily_change.rolling(window=window_ker).sum()

# 3. Final KER calculation (with epsilon to prevent division by zero)
daily_df['KER_22D'] = net_change / (path_length + 1e-8)


# =================================================================================================== Directional Drift / Daily Return


window_z_Drift = 120

# 1. Daily Drift (Close-to-Close Log Return)
# Full day log return including gaps and entire session
daily_df['Log_Return'] = np.log(daily_df['Close'] / daily_df['Close'].shift(1))


# Rolling 90-day Z-Score to normalize drift relative to current regime
mean_drift = daily_df['Log_Return'].rolling(window=window_z_Drift).mean()
std_drift = daily_df['Log_Return'].rolling(window=window_z_Drift).std()
daily_df['Drift_ZScore'] = ((daily_df['Log_Return'] - mean_drift) / (std_drift + 1e-8)).clip(-3.5, 3.5)



# 2. Intraday Drift (Open-to-Close Body Drift)
# Today's candle body return (pure intraday directional move)
daily_df['Log_Body_Drift'] = np.log(daily_df['Close'] / daily_df['Open'])



mean_body = daily_df['Log_Body_Drift'].rolling(window=window_z_Drift).mean()
std_body = daily_df['Log_Body_Drift'].rolling(window=window_z_Drift).std()
daily_df['Intraday_Drift_ZScore'] = ((daily_df['Log_Body_Drift'] - mean_body) / (std_body + 1e-8)).clip(-3.5, 3.5)



# =================================================================================================== Efficiency (Absolute-Signed)

window_z_Efficiency = 120


# a) Absolute Efficiency (0 to 1+ range)
# Measures how clean the candle is (Marubozu ~1, Doji ~0)
# Useful for range prediction: high efficiency = strong directional move
daily_df['Candle_Efficiency_Abs'] = (
    daily_df['Log_Body_Drift'].abs() / (daily_df['RS_Vol_1D'] + 1e-8)
).clip(0.0, 1.5)


# b) Signed Efficiency + Rolling Z-Score (90-day)
# Measures buyer (+) vs seller (-) dominance relative to the 90-day vol regime
daily_df['Candle_Efficiency_Signed'] = daily_df['Log_Body_Drift'] / (daily_df['RS_Vol_1D'] + 1e-8)


# Candle_Efficiency_Signed_ZScore
mean_eff = daily_df['Candle_Efficiency_Signed'].rolling(window=window_z_Efficiency).mean()
std_eff = daily_df['Candle_Efficiency_Signed'].rolling(window=window_z_Efficiency).std()

daily_df['Candle_Efficiency_Signed_ZScore'] = (
    (daily_df['Candle_Efficiency_Signed'] - mean_eff) / (std_eff + 1e-8)
).clip(-3.5, 3.5)


# =================================================================================================== Intraday Downside , Upside Semivariance


window_z_semivar = 120

# Step 1: 1-hour log returns
hourly_df["Log_Ret_1H"] = np.log(hourly_df["Close"] / hourly_df["Close"].shift(1))

# Step 2: Separate squared upside and downside returns
hourly_df["Ret_Pos_Sq"] = np.where(hourly_df["Log_Ret_1H"] > 0, hourly_df["Log_Ret_1H"] ** 2, 0.0)
hourly_df["Ret_Neg_Sq"] = np.where(hourly_df["Log_Ret_1H"] < 0, hourly_df["Log_Ret_1H"] ** 2, 0.0)



# Step 3: Aggregate to daily level and merge into daily_df
sv_daily = hourly_df.resample("D").agg(
    {"Ret_Pos_Sq": "sum", "Ret_Neg_Sq": "sum"}
).rename(columns={"Ret_Pos_Sq": "SV_Plus", "Ret_Neg_Sq": "SV_Minus"})

daily_df = daily_df.join(sv_daily)



# Step 4: Feature Engineering

# 4.1. Semivariance Asymmetry Ratio (-1 to +1)
# Negative = downside dominance, Positive = upside dominance
daily_df["SV_Asymmetry"] = (daily_df["SV_Plus"] - daily_df["SV_Minus"]) / (
    daily_df["SV_Plus"] + daily_df["SV_Minus"] + 1e-8
)

mean_asym = daily_df["SV_Asymmetry"].rolling(window=window_z_semivar).mean()
std_asym = daily_df["SV_Asymmetry"].rolling(window=window_z_semivar).std()

daily_df["SV_Asymmetry_ZScore"] = (
    (daily_df["SV_Asymmetry"] - mean_asym) / (std_asym + 1e-8)
).clip(-3.5, 3.5)



# 4.2. Log Semivariance Ratio (symmetric, Gaussian-friendly)
# Log of upside vs downside variance ratio
daily_df["Log_SV_Ratio"] = np.log(
    (daily_df["SV_Plus"] + 1e-8) / (daily_df["SV_Minus"] + 1e-8)
)

mean_log_ratio = daily_df["Log_SV_Ratio"].rolling(window=window_z_semivar).mean()
std_log_ratio = daily_df["Log_SV_Ratio"].rolling(window=window_z_semivar).std()

daily_df["Log_SV_Ratio_ZScore"] = (
    (daily_df["Log_SV_Ratio"] - mean_log_ratio) / (std_log_ratio + 1e-8)
).clip(-3.5, 3.5)






# ============================================================================================== Realized Skewness , Realized Kurtosis


window_z_SK = 120

# 1. HIGHER MOMENTS ON 1-HOUR DATA (hourly_df)

# 1.1. 1-hour log returns
hourly_df["Log_Ret_1H"] = np.log(hourly_df["Close"] / hourly_df["Close"].shift(1))

# 1.2. Powers of 1-hour returns (2, 3, 4)
hourly_df["Ret_Sq_1H"]   = hourly_df["Log_Ret_1H"] ** 2
hourly_df["Ret_Cube_1H"] = hourly_df["Log_Ret_1H"] ** 3
hourly_df["Ret_Quad_1H"] = hourly_df["Log_Ret_1H"] ** 4



# 2. AGGREGATE TO DAILY LEVEL (Barndorff-Nielsen standard formulas)

# Sum of powers and active bar count per day
moments_daily = hourly_df.resample("D").agg(
    {
        "Ret_Sq_1H": ["sum", "count"],
        "Ret_Cube_1H": "sum",
        "Ret_Quad_1H": "sum",
    }
)

# Flatten multi-level column names
moments_daily.columns = ["Sum_R2", "N_count", "Sum_R3", "Sum_R4"]

# Standard Amaya et al. (2015) formulas for High-Frequency Realized Moments
# Realized Skewness = (sqrt(N) * sum(r^3)) / (sum(r^2)^(3/2))
moments_daily["Realized_Skew"] = (
    np.sqrt(moments_daily["N_count"]) * moments_daily["Sum_R3"]
) / (np.power(moments_daily["Sum_R2"], 1.5) + 1e-8)

# Realized Kurtosis = (N * sum(r^4)) / (sum(r^2)^2)
moments_daily["Realized_Kurt"] = (
    moments_daily["N_count"] * moments_daily["Sum_R4"]
) / (np.square(moments_daily["Sum_R2"]) + 1e-8)

# Merge into daily_df
daily_df = daily_df.join(
    moments_daily[["Realized_Skew", "Realized_Kurt"]]
)



# 3. STANDARDIZATION FOR REGRESSION INPUT

# 3.1. Rolling Z-Score for Realized Skewness (naturally symmetric around zero)
mean_skew = daily_df["Realized_Skew"].rolling(window=window_z_SK).mean()
std_skew  = daily_df["Realized_Skew"].rolling(window=window_z_SK).std()

daily_df["Realized_Skew_ZScore"] = (
    (daily_df["Realized_Skew"] - mean_skew) / (std_skew + 1e-8)
).clip(-3.5, 3.5)



# 3.2. Log transform + Rolling Z-Score for Realized Kurtosis
# (Kurtosis is positive and highly right-skewed, so log is applied first)
daily_df["Log_Realized_Kurt"] = np.log(daily_df["Realized_Kurt"] + 1e-8)

mean_kurt = daily_df["Log_Realized_Kurt"].rolling(window=window_z_SK).mean()
std_kurt  = daily_df["Log_Realized_Kurt"].rolling(window=window_z_SK).std()

daily_df["Realized_Kurt_ZScore"] = (
    (daily_df["Log_Realized_Kurt"] - mean_kurt) / (std_kurt + 1e-8)
).clip(-3.5, 3.5)


# FEATURE (Realized_Kurt_ZScore, Realized_Skew_ZScore)






# =================================================================================================== Bipower Variation (BPV)


# CONFIGURATION
window_z_jp = 120  # Rolling window for Z-Score standardization



# 1. COMPUTATIONS ON 1-HOUR DATA (hourly_df)

# 1.1. 1-hour log returns
hourly_df["Log_Ret_1H"] = np.log(hourly_df["Close"] / hourly_df["Close"].shift(1))

# 1.2. Squared 1-hour returns (for Realized Volatility)
hourly_df["Ret_Sq_1H"] = hourly_df["Log_Ret_1H"] ** 2

# 1.3. Product of absolute consecutive returns (Bipower Variation base)
hourly_df["BPV_Prod_1H"] = hourly_df["Log_Ret_1H"].abs() * hourly_df["Log_Ret_1H"].abs().shift(1)




# 2. AGGREGATE TO DAILY LEVEL AND MERGE WITH daily_df

# (pi/2) is the standard scaling factor from Barndorff-Nielsen & Shephard
bpv_daily = hourly_df.resample("D").agg(
    {
        "Ret_Sq_1H": "sum",
        "BPV_Prod_1H": lambda x: (np.pi / 2.0) * x.sum(),
    }
).rename(
    columns={
        "Ret_Sq_1H": "Realized_RV",
        "BPV_Prod_1H": "Realized_BPV",
    }
)

daily_df = daily_df.join(bpv_daily)





# 3. FEATURE ENGINEERING (Standardized for Regression)

# 3.1. Jump Component (positive part of RV - BPV)
daily_df["Jump_Component"] = (
    daily_df["Realized_RV"] - daily_df["Realized_BPV"]
).clip(lower=0.0)


# 3.2. Jump Ratio — share of jumps in total daily variance (0 to 1)
daily_df["Jump_Ratio"] = daily_df["Jump_Component"] / (
    daily_df["Realized_RV"] + 1e-8
)
daily_df["Jump_Ratio"] = daily_df["Jump_Ratio"].clip(0.0, 1.0)



# 3.3. Rolling Z-Score of Jump Ratio (90-day)
mean_jump = daily_df["Jump_Ratio"].rolling(window=window_z_jp).mean()
std_jump = daily_df["Jump_Ratio"].rolling(window=window_z_jp).std()

daily_df["Jump_Ratio_ZScore"] = (
    (daily_df["Jump_Ratio"] - mean_jump) / (std_jump + 1e-8)
).clip(-3.5, 3.5)



# 3.4. Log Continuous Ratio — symmetric, Gaussian-friendly feature
# Measures the share of diffusive (non-jump) volatility
daily_df["Log_Continuous_Ratio"] = np.log(
    (daily_df["Realized_BPV"] + 1e-8) / (daily_df["Realized_RV"] + 1e-8)
)

mean_cont = daily_df["Log_Continuous_Ratio"].rolling(window=window_z_jp).mean()
std_cont = daily_df["Log_Continuous_Ratio"].rolling(window=window_z_jp).std()


daily_df["Continuous_Ratio_ZScore"] = (
    (daily_df["Log_Continuous_Ratio"] - mean_cont) / (std_cont + 1e-8)
).clip(-3.5, 3.5)


# FEATURE (Continuous_Ratio_ZScore, Jump_Ratio_ZScore)



# =================================================================================================== Range_CV_22D



window_cv = 22      # Monthly window for volatility instability
window_z_cv = 120      # Rolling Z-Score standardization window

# 1. VOL-OF-VOL / RANGE CV CALCULATION

# 1.1. Rolling 22-day standard deviation of daily volatility
std_vol_22d = daily_df["RS_Vol_1D"].rolling(window=window_cv).std()

# 1.2. Rolling 22-day mean of daily volatility
mean_vol_22d = daily_df["RS_Vol_1D"].rolling(window=window_cv).mean()

# 1.3. CV ratio = std / mean (scale-free)
daily_df["Range_CV_22D"] = std_vol_22d / (mean_vol_22d + 1e-8)



# 2. STANDARDIZATION FOR REGRESSION INPUT


# 2.1. Log transform to correct right-skewness
daily_df["Log_Range_CV"] = np.log(daily_df["Range_CV_22D"] + 1e-8)


# 2.2. Rolling 90-day Z-Score
mean_log_cv = daily_df["Log_Range_CV"].rolling(window=window_z_cv).mean()
std_log_cv = daily_df["Log_Range_CV"].rolling(window=window_z_cv).std()


daily_df["Vol_of_Vol_ZScore"] = (
    (daily_df["Log_Range_CV"] - mean_log_cv) / (std_log_cv + 1e-8)
).clip(-3.5, 3.5)


# =================================================================================================== ROLLING DETRENDED FLUCTUATION ANALYSIS (DFA) SCALING EXPONENT


def compute_dfa_metrics(series, scales=(4, 8, 12, 16, 24, 32)):
    """
    Computes DFA scaling exponent (alpha) and Goodness of Fit (R^2)
    Safe against missing scales and prevents division by zero.
    """
    series = np.asarray(series, dtype=float)
    if np.isnan(series).any() or len(series) < max(scales):
        return np.nan, np.nan

    # Remove mean and integrate
    y = np.cumsum(series - np.mean(series))
    
    valid_scales = []
    fluctuations = []

    for s in scales:
        n_segments = len(y) // s
        if n_segments < 1:
            continue

        segments = y[:n_segments * s].reshape(n_segments, s)
        x = np.arange(s)

        # DFA-1 Detrending
        poly = np.polyfit(x, segments.T, deg=1)
        trend = np.polyval(poly, x[:, None]).T

        rms = np.sqrt(np.mean((segments - trend) ** 2))

        if np.isfinite(rms) and rms > 0:
            valid_scales.append(s)
            fluctuations.append(rms)

    # Require at least 3 valid scales for robust linear fit
    if len(fluctuations) < 3:
        return np.nan, np.nan

    log_scales = np.log(valid_scales)
    log_fluct = np.log(fluctuations)

    # Linear Regression log(F(s)) ~ log(s)
    poly = np.polyfit(log_scales, log_fluct, deg=1)
    alpha = poly[0]

    # Calculate R-squared (Goodness of Fit)
    y_pred = poly[0] * log_scales + poly[1]
    ss_res = np.sum((log_fluct - y_pred) ** 2)
    ss_tot = np.sum((log_fluct - np.mean(log_fluct)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return alpha, r2


# =========================================================================
# APPLYING TO DAILY VOLATILITY (e.g., 90-day window)
# =========================================================================

# Example application on Log_RS_Vol_1D:
dfa_results = daily_df["Log_RS_Vol_1D"].rolling(window=90).apply(
    lambda w: compute_dfa_metrics(w)[0], raw=True
)
dfa_r2_results = daily_df["Log_RS_Vol_1D"].rolling(window=90).apply(
    lambda w: compute_dfa_metrics(w)[1], raw=True
)

daily_df["f_dfa_alpha_90d"] = dfa_results
daily_df["f_dfa_r2_90d"] = dfa_r2_results



# =================================================================================================== Permutation Entropy


def compute_permutation_entropy(series, m=3, delay=1):


    """Computes Normalized Permutation Entropy (PE) over a 1D window.

    PE = 1.0 -> Complete Chaos / Noise (Chop) PE < 0.6 -> Deterministic Order
    (Trend / Compression)
    """
    
    series = np.asarray(series, dtype=float)
    n = len(series)

    # Required minimum points for dimension m
    if n < (m - 1) * delay + 1 or np.isnan(series).any():
        return np.nan

    # 1. Extract ordinal pattern vectors
    patterns = []
    for i in range(n - (m - 1) * delay):
        # Extract sub-window
        window = series[i : i + m * delay : delay]
        # Get ordinal ranks (permutation pattern)
        pattern = tuple(np.argsort(window))
        patterns.append(pattern)

    if not patterns:
        return np.nan

    # 2. Compute probabilities of each pattern
    counts = Counter(patterns)
    total_patterns = len(patterns)

    # 3. Shannon Entropy
    pe = 0.0
    for count in counts.values():
        p = count / total_patterns
        pe -= p * math.log2(p)

    # 4. Normalize by max entropy log2(m!) so range is bounded [0, 1]
    max_entropy = math.log2(math.factorial(m))
    return pe / max_entropy


# =========================================================================
# APPLYING TO DAILY CLOSE / RETURNS (30-day rolling window)
# =========================================================================

# Applied to daily Close price to capture price pattern complexity
daily_df["f_pe_m3_w30"] = (
    daily_df["Close"]
    .rolling(window=30)
    .apply(lambda w: compute_permutation_entropy(w, m=3, delay=1), raw=True)
)



# =================================================================================================== Fractional Differencing


def get_ffd_weights(d: float, thres: float = 1e-4) -> np.ndarray:
    """Calculates memory weights for Fractional Differencing (FFD)."""
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])  # Reverse weights for dot product


def frac_diff_ffd(
    series: pd.Series, d: float, thres: float = 1e-4
) -> pd.Series:
    """Applies Fixed-Width Window Fractional Differencing to a Series.

    Parameters:
    - series: Raw price series (e.g. daily_df['Close'])
    - d: Fractional degree (typically between 0.2 and 0.6 for financial data)
    - thres: Weight loss threshold to fix window width
    """
    weights = get_ffd_weights(d, thres)
    width = len(weights)

    # Compute dot product over rolling windows
    res = {}
    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1 : i + 1]
        if window.isnull().any():
            continue
        res[series.index[i]] = np.dot(weights, window)

    return pd.Series(res, name=f"f_fracdiff_d{d:.2f}")


# =========================================================================
# APPLYING TO DAILY CLOSE PRICE
# =========================================================================

# Usually d=0.35 to 0.45 works best for Forex pairs (e.g., EUR/USD)
# It retains trend/support/resistance memory while passing ADF stationarity test
daily_df["f_close_fracdiff_d40"] = frac_diff_ffd(
    daily_df["Close"], d=0.40, thres=1e-4
)


# =================================================================================================== CUSUM (Cumulative Sum)

# def compute_cusum_features(
#     daily_df: pd.DataFrame,
#     hourly_df: pd.DataFrame,
#     threshold_std_mult: float = 1.5,
# ) -> pd.DataFrame:
#     """Computes CUSUM structural break metrics for daily and intraday levels."""
#     # 1. Daily Level CUSUM Drift
#     daily_ret = np.log(daily_df["Close"] / daily_df["Close"].shift(1)).fillna(0)
#     vol_target = daily_ret.rolling(30).std().bfill()

#     s_pos, s_neg = 0.0, 0.0
#     drift_series = []

#     for r, v in zip(daily_ret, vol_target):
#         h = threshold_std_mult * v if v > 0 else 1e-4
#         s_pos = max(0.0, s_pos + r)
#         s_neg = min(0.0, s_neg + r)

#         max_drift = max(s_pos, abs(s_neg)) / h
#         drift_series.append(max_drift)

#         if s_pos >= h:
#             s_pos = 0.0
#         if abs(s_neg) >= h:
#             s_neg = 0.0

#     daily_df["f_cusum_drift_daily"] = drift_series

#     # 2. Intraday Event Counter from Hourly Data
#     hourly_ret = np.log(
#         hourly_df["Close"] / hourly_df["Close"].shift(1)
#     ).fillna(0)
#     hourly_vol = hourly_ret.rolling(24).std().bfill()

#     h_events = []
#     s_p, s_n = 0.0, 0.0

#     for r, v in zip(hourly_ret, hourly_vol):
#         h = threshold_std_mult * v if v > 0 else 1e-4
#         s_p = max(0.0, s_p + r)
#         s_n = min(0.0, s_n + r)

#         event = 0
#         if s_p >= h:
#             event = 1
#             s_p = 0.0
#         elif abs(s_n) >= h:
#             event = 1
#             s_n = 0.0

#         h_events.append(event)

#     hourly_df["cusum_event"] = h_events

#     daily_events = hourly_df.groupby(hourly_df.index.date)["cusum_event"].sum()
#     daily_events.index = pd.to_datetime(daily_events.index)

#     daily_df["f_cusum_events_24h"] = daily_df.index.map(daily_events).fillna(0)

#     return daily_df

# ==============================================================================================================================================================


# 1. Define the list of selected features available in daily_df
selected_features = [
    "f_close_fracdiff_d40",
    "f_pe_m3_w30",
    "Realized_Kurt_ZScore",
    "Realized_Skew_ZScore",
    "f_dfa_alpha_90d",
    "f_dfa_r2_90d",
    "Vol_of_Vol_ZScore",
    "Vol_Divergence_1_5",
    "Vol_Term_Structure_5_22",
    "Log_RS_Vol_1D_ZScore",
    "day_of_week",
    "SV_Asymmetry_ZScore",
    "f_quarter_progress",
    "f_month_progress"


    # Add your other feature column names here
]

# 2. Add the Target column if needed (e.g., target_daily_range)
# selected_features.append('target_range')

# 3. Extract selected features into a new DataFrame
features_df = daily_df[selected_features].copy()

# 4. Drop NaN values caused by rolling window lookbacks
features_df = features_df.dropna()

# 5. Export to CSV file preserving the datetime index
output_filename = "selected_features.csv"
features_df.to_csv(output_filename, index=True)

print(
    f"Successfully exported {len(selected_features)} features with shape {features_df.shape} to '{output_filename}'."
)



