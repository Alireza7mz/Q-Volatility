



# Data processing and numerical computations
import numpy as np
import pandas as pd

# Plotting and reporting
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing and Ridge modeling
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# Multicollinearity check (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Model evaluation metrics
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    accuracy_score,
    confusion_matrix,
    classification_report,
    max_error,
    silhouette_score
)




from data_loader import get_cleaned_data

# Daily Data
daily_df = get_cleaned_data(timeframe="1D")
df = get_cleaned_data(timeframe="1h")




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
df["Log_Ret_1H"] = np.log(df["Close"] / df["Close"].shift(1))

# Step 2: Separate squared upside and downside returns
df["Ret_Pos_Sq"] = np.where(df["Log_Ret_1H"] > 0, df["Log_Ret_1H"] ** 2, 0.0)
df["Ret_Neg_Sq"] = np.where(df["Log_Ret_1H"] < 0, df["Log_Ret_1H"] ** 2, 0.0)



# Step 3: Aggregate to daily level and merge into daily_df
sv_daily = df.resample("D").agg(
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

# 1. HIGHER MOMENTS ON 1-HOUR DATA (df)

# 1.1. 1-hour log returns
df["Log_Ret_1H"] = np.log(df["Close"] / df["Close"].shift(1))

# 1.2. Powers of 1-hour returns (2, 3, 4)
df["Ret_Sq_1H"]   = df["Log_Ret_1H"] ** 2
df["Ret_Cube_1H"] = df["Log_Ret_1H"] ** 3
df["Ret_Quad_1H"] = df["Log_Ret_1H"] ** 4



# 2. AGGREGATE TO DAILY LEVEL (Barndorff-Nielsen standard formulas)

# Sum of powers and active bar count per day
moments_daily = df.resample("D").agg(
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



# 1. COMPUTATIONS ON 1-HOUR DATA (df)

# 1.1. 1-hour log returns
df["Log_Ret_1H"] = np.log(df["Close"] / df["Close"].shift(1))

# 1.2. Squared 1-hour returns (for Realized Volatility)
df["Ret_Sq_1H"] = df["Log_Ret_1H"] ** 2

# 1.3. Product of absolute consecutive returns (Bipower Variation base)
df["BPV_Prod_1H"] = df["Log_Ret_1H"].abs() * df["Log_Ret_1H"].abs().shift(1)




# 2. AGGREGATE TO DAILY LEVEL AND MERGE WITH daily_df

# (pi/2) is the standard scaling factor from Barndorff-Nielsen & Shephard
bpv_daily = df.resample("D").agg(
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


# =================================================================================================== Multi Reggression OLS



# 1. TARGET VARIABLE Y (Predict t+1 volatility)

# Target: tomorrow's Rogers-Satchell log-volatility
daily_df["Target_Y"] = np.log(daily_df["RS_Vol_1D"] + 1e-8).shift(-1)






# Final feature list (Variance + Memory categories)
feature_cols = [
    # Variance category
    "Log_RS_Vol_1D_ZScore",
    "Candle_Efficiency_Abs",
    "Jump_Ratio_ZScore",
    "SV_Asymmetry_ZScore",
    "Drift_ZScore",

    # Memory category
    "Vol_Divergence_1_5",
    "Vol_Term_Structure_5_22",
    "Log_RS_Vol_22D_ZScore",
    "Vol_of_Vol_ZScore",
    "Log_RS_Autocorr1",
    "KER_22D",
]

# Drop last row (Target_Y is NaN for the last day) and clean initial NaNs
clean_df = daily_df.dropna(subset=feature_cols + ["Target_Y"]).copy()

X = clean_df[feature_cols]
y = clean_df["Target_Y"]






# 2. WALK-FORWARD VALIDATION (Expanding Window)

min_train_size = 252  # Minimum training size: 1 trading year
predictions = []
actuals = []
dates = []
coefficients = []

# Alpha range for Ridge
alphas = np.logspace(-3, 3, 50)

# Walk-Forward loop (train on past, predict exactly 1 day ahead)
for i in range(min_train_size, len(clean_df)):
    # Training data: from start to day i-1
    X_train = X.iloc[:i]
    y_train = y.iloc[:i]

    # Test data: exactly day i (out-of-sample)
    X_test = X.iloc[i : i + 1]
    y_test = y.iloc[i]

    # Ridge with internal CV on training data
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_train, y_train)

    # Out-of-sample prediction
    y_pred = model.predict(X_test)[0]

    # Store results
    predictions.append(y_pred)
    actuals.append(y_test)
    dates.append(clean_df.index[i])
    coefficients.append(model.coef_)

# Results DataFrame
results_df = pd.DataFrame(
    {"Actual": actuals, "Predicted": predictions}, index=dates
)




# 3. OUT-OF-SAMPLE EVALUATION METRICS

oos_r2 = r2_score(results_df["Actual"], results_df["Predicted"])
oos_rmse = np.sqrt(
    mean_squared_error(results_df["Actual"], results_df["Predicted"])
)
oos_mae = np.mean(np.abs(results_df["Actual"] - results_df["Predicted"]))

# Directional accuracy (vol expansion/contraction)
actual_diff = np.diff(results_df["Actual"])
pred_diff = np.diff(results_df["Predicted"])
dir_accuracy = np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100


# print("=== Out-of-Sample Evaluation Results (Walk-Forward) ===")
# print(f"OOS R-Squared (R²): {oos_r2:.4f}")
# print(f"OOS RMSE:           {oos_rmse:.4f}")
# print(f"OOS MAE:            {oos_mae:.4f}")
# print(f"Directional Acc:    {dir_accuracy:.2f}%")





# # =================================================================================================== CALCULATE MODEL ERROR IN PIPS

# # 1. Convert actual and predicted values to pips
# close_prices = clean_df.loc[results_df["Actual"].index, "Close"]

# actual_pips = np.exp(results_df["Actual"]) * close_prices * 10000
# predicted_pips = np.exp(results_df["Predicted"]) * close_prices * 10000

# # 2. MAE in pips
# mae_pips = np.mean(np.abs(actual_pips - predicted_pips))

# # 3. RMSE in pips
# rmse_pips = np.sqrt(np.mean((actual_pips - predicted_pips) ** 2))

# # 4. Corrected R2 in pips space
# r2_pips_corrected = r2_score(actual_pips, predicted_pips)

# print(
#     f"Mean Absolute Error (MAE) in pips: {mae_pips:.2f} Pips"
# )
# print(f"Root Mean Squared Error (RMSE) in pips: {rmse_pips:.2f} Pips")
# print(f"R-Squared (Corrected, Pips Space): {r2_pips_corrected:.4f}")


# # Calculate Median Absolute Error
# medae_pips = np.median(np.abs(actual_pips - predicted_pips))

# print(f"Mean Absolute Error (MAE): {mae_pips:.2f} Pips")
# print(f"Median Absolute Error (MedAE - normal days): {medae_pips:.2f} Pips")






# ---------------------------------------------------------
# 4. EXPORT CLEAN LINEAR MODEL PREDICTIONS FOR PIPELINE
# ---------------------------------------------------------

# 1. Create output dataframe with standard columns
df_linear_export = pd.DataFrame(
    {
        "pred_linear": results_df[
            "Predicted"
        ],  # OOS predictions from linear model for next-day volatility
        "target_actual": results_df[
            "Actual"
        ],  # Actual target values (for error calculation / benchmarking)
    },
    index=results_df.index,
)



# 2. Convert datetime index to timestamp column
df_linear_export = df_linear_export.reset_index()
df_linear_export = df_linear_export.rename(columns={"index": "timestamp"})



# 3. Save clean output to CSV file
df_linear_export.to_csv("linear_model_preds.csv", index=False)



print("✅ Linear model output file successfully saved:")
print(f"Number of predicted records: {len(df_linear_export)}")
print(df_linear_export.head())

