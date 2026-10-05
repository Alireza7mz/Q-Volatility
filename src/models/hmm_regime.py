

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from hmmlearn.hmm import GaussianHMM
from data_loader import get_cleaned_data




# =========================================================================
# STEP 0: LOAD DATA
# =========================================================================


daily_df = get_cleaned_data(timeframe="1D")


# =========================================================================
# STEP 1: CALCULATE CAUSAL FEATURES
# =========================================================================
# f1: Rogers-Satchell Volatility Level (5-day rolling horizon)
u = np.log(daily_df['High'] / daily_df['Open'])
d = np.log(daily_df['Low']  / daily_df['Open'])
c = np.log(daily_df['Close'] / daily_df['Open'])

rs_var_1d   = np.maximum(u * (u - c) + d * (d - c), 0)
rs_vol_1d   = np.sqrt(rs_var_1d)
rs_vol_5d   = np.sqrt(rs_var_1d.rolling(window=5).mean())
rs_vol_22d  = np.sqrt(rs_var_1d.rolling(window=22).mean())

log_vol_5d  = np.log(rs_vol_5d + 1e-8)
log_vol_22d = np.log(rs_vol_22d + 1e-8)

# f2.1: Multi-horizon Variance Ratio
log_ret_1      = np.log(daily_df['Close'] / daily_df['Close'].shift(1))
log_ret_5      = np.log(daily_df['Close'] / daily_df['Close'].shift(5))
var_1d_rolling = log_ret_1.rolling(window=22).var()
var_5d_rolling = log_ret_5.rolling(window=22).var()

f2_var_ratio    = var_5d_rolling / (5 * var_1d_rolling + 1e-12)

# f2.2: Volatility Momentum / Relative Volatility State (log(vol_5d / vol_22d))
f2_vol_momentum = log_vol_5d - log_vol_22d

# f3: Volatility Instability (std of log-volatility over 22 days)
vol_instability_raw = log_vol_5d.rolling(window=22).std()
f3_vol_instability  = np.log(vol_instability_raw + 1e-8)

# f6: Modified Kaufman Efficiency Ratio (mKER) over 10-day window
net_change   = (daily_df['Close'] - daily_df['Close'].shift(10)).abs()
daily_change = (daily_df['Close'] - daily_df['Close'].shift(1)).abs()
path_length  = daily_change.rolling(window=10).sum()
f6_mker      = net_change / (path_length + 1e-12)

# Assemble feature matrix
features_df = pd.DataFrame({
    'f1_log_vol_5d': log_vol_5d,
    'f2_var_ratio': f2_var_ratio,
    'f2_vol_momentum': f2_vol_momentum,
    'f3_vol_instability': f3_vol_instability,
    'f6_mker': f6_mker
}, index=daily_df.index).dropna()




# =========================================================================
# STEP 2: WALK-FORWARD HMM WITH CAUSAL ONLINE FILTERING & PROBABILITIES
# =========================================================================


train_window = 252  # 1-year rolling historical window
n_samples = len(features_df)

walk_forward_regimes = np.full(n_samples, np.nan)
prob_regime_0 = np.full(n_samples, np.nan)  # P(Low Volatility / Compression)
prob_regime_1 = np.full(n_samples, np.nan)  # P(Medium Volatility / Transition)
prob_regime_2 = np.full(n_samples, np.nan)  # P(High Volatility / Expansion)


feature_cols = ['f1_log_vol_5d', 'f2_var_ratio', 'f2_vol_momentum', 'f3_vol_instability', 'f6_mker']



for t in range(train_window, n_samples):
    train_data = features_df.iloc[t - train_window : t][feature_cols]
    current_bar = features_df.iloc[t : t + 1][feature_cols]
    


    # Quantile mapping with n_quantiles=50 for tail stability on 252-day window
    wf_scaler = QuantileTransformer(n_quantiles=50, output_distribution='normal', random_state=42)
    train_scaled = wf_scaler.fit_transform(train_data)
    current_scaled = wf_scaler.transform(current_bar)
    


    # Fit multiple HMMs to find best initialization (Highest Log-Likelihood)
    best_score = -np.inf
    best_hmm = None
    
    for seed in range(5):
        candidate_hmm = GaussianHMM(
            n_components=3, 
            covariance_type="diag", 
            n_iter=300, 
            tol=1e-3, 
            random_state=42 + seed
        )
        candidate_hmm.fit(train_scaled)
        score = candidate_hmm.score(train_scaled)
        
        if score > best_score:
            best_score = score
            best_hmm = candidate_hmm
            
    wf_hmm = best_hmm
    


    # Fix Label Flipping: Sort states by f1 volatility in training data
    train_preds = wf_hmm.predict(train_scaled)
    raw_f1_means = train_data.groupby(train_preds)['f1_log_vol_5d'].mean()
    sorted_states = raw_f1_means.sort_values().index
    state_map = {old_st: new_st for new_st, old_st in enumerate(sorted_states)}
    


    # Causal Online Filtering: Attach current observation to historical window
    extended_scaled = np.vstack([train_scaled, current_scaled])
    


    # Compute endpoint posterior probabilities P(S_t | X_1:t) without future leakage
    posteriors = wf_hmm.predict_proba(extended_scaled)
    raw_current_probs = posteriors[-1]
    


    # Map probabilities to sorted states
    sorted_probs = np.zeros(3)
    for raw_idx, sorted_idx in state_map.items():
        sorted_probs[sorted_idx] = raw_current_probs[raw_idx]
        
    prob_regime_0[t] = sorted_probs[0]
    prob_regime_1[t] = sorted_probs[1]
    prob_regime_2[t] = sorted_probs[2]
    walk_forward_regimes[t] = np.argmax(sorted_probs)



# Attach outputs to dataframe
features_df['wf_regime'] = walk_forward_regimes
features_df['prob_regime_0'] = prob_regime_0
features_df['prob_regime_1'] = prob_regime_1
features_df['prob_regime_2'] = prob_regime_2


features_clean = features_df.dropna(subset=['wf_regime']).copy()
features_clean['wf_regime'] = features_clean['wf_regime'].astype(int)





# ---------------------------------------------------------
# EXPORT CLEAN HMM REGIME OUTPUTS FOR PIPELINE
# ---------------------------------------------------------

# 1. Select main HMM output columns
hmm_cols = ['wf_regime', 'prob_regime_0', 'prob_regime_1', 'prob_regime_2']

# 2. Convert datetime index to timestamp column and build output dataframe
df_hmm_export = features_clean[hmm_cols].reset_index()

# Rename date column to timestamp if needed
if 'timestamp' not in df_hmm_export.columns:
    df_hmm_export = df_hmm_export.rename(
        columns={df_hmm_export.columns[0]: 'timestamp'}
    )

# 3. Save to CSV file
df_hmm_export.to_csv('regimes.csv', index=False)

print('✅ HMM regime file successfully saved:')
print(f'Number of calculated records: {len(df_hmm_export)}')
print(df_hmm_export.head())






# # =========================================================================
# # STEP 3: INSPECT & SAVE RESULTS
# # =========================================================================


# print("=== Out-of-Sample Regime Distribution ===")
# print(features_clean['wf_regime'].value_counts(normalize=True).round(3))


# print("\n=== Feature Averages per Sorted Regime ===")
# print(features_clean.groupby('wf_regime')[feature_cols].mean().round(3))


# # Save processed features and probabilities for downstream regression modeling
# features_clean.to_csv("regimes.csv")
# print("\n[SUCCESS] Exported features_with_regimes.csv successfully.")



