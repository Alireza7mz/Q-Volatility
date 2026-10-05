


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import pandas as pd


from data_loader import load_merged_data


# 1. Load unified DataFrame
df = load_merged_data()

# 2. Define target variable and separate features
TARGET_COL = "target_actual"  # or whatever name is used in the linear file

y = df[TARGET_COL]

# Drop target column and any other non-feature columns from X
cols_to_drop = [TARGET_COL]
X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])



#---------------------------------------- Walk-Forward Validation



INITIAL_TRAIN_SIZE = 500  # ~2 years of initial training data
STEP_SIZE = 1  # Day-by-day out-of-sample predictions

predictions = []
actuals = []
linear_preds = []
dates = []

# Hyperparameter settings (strict overfitting control on financial data)
xgb_params = {
    "n_estimators": 100,
    "max_depth": 3,  # Shallow trees to prevent noise memorization
    "learning_rate": 0.03,  # Low learning rate for gradual learning
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

print(f"🚀 Starting Walk-Forward Validation on {len(df) - INITIAL_TRAIN_SIZE} days...")

for i in range(INITIAL_TRAIN_SIZE, len(df), STEP_SIZE):
    X_train, y_train = X.iloc[:i], y.iloc[:i]
    X_test = X.iloc[i : i + STEP_SIZE]
    y_test = y.iloc[i : i + STEP_SIZE]

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    predictions.extend(preds)
    actuals.extend(y_test.values)
    linear_preds.extend(X_test["pred_linear"].values)
    dates.extend(X_test.index)

# 3. Performance evaluation and comparison
results_df = pd.DataFrame(
    {
        "Actual": actuals,
        "XGB_Pred": predictions,
        "Linear_Pred": linear_preds,
    },
    index=dates,
)

xgb_rmse = np.sqrt(mean_squared_error(results_df["Actual"], results_df["XGB_Pred"]))
linear_rmse = np.sqrt(mean_squared_error(results_df["Actual"], results_df["Linear_Pred"]))

xgb_mae = mean_absolute_error(results_df["Actual"], results_df["XGB_Pred"])
linear_mae = mean_absolute_error(results_df["Actual"], results_df["Linear_Pred"])






#---------------------------------------- print - plt




print("\n" + "="*40)
print("📊 OUT-OF-SAMPLE EVALUATION RESULTS")
print("="*40)
print(f"XGBoost RMSE: {xgb_rmse:.5f} | MAE: {xgb_mae:.5f}")
print(f"Linear  RMSE: {linear_rmse:.5f} | MAE: {linear_mae:.5f}")
print("="*40)

# 4. Feature importance on full dataset
final_model = xgb.XGBRegressor(**xgb_params)
final_model.fit(X, y)

importance = pd.Series(final_model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=True)

# plt.figure(figsize=(10, 6))
# importance.plot(kind="barh", color="skyblue", edgecolor="black")
# plt.title("XGBoost Feature Importance (Gain)")
# plt.xlabel("Importance Score")
# plt.tight_layout()
# plt.savefig("feature_importance.png")
# plt.show()

# # 5. Save final predictions
results_df.to_csv("xgb_final_predictions.csv")
print("✅ Predictions saved to 'xgb_final_predictions.csv'.")




# ==============================================================================
# 📊 DETAILED OUT-OF-SAMPLE EVALUATION & COMPARISON
# ==============================================================================
from scipy.stats import pearsonr, spearmanr

def evaluate_models(df, y_col="Actual", lin_col="Linear_Pred", xgb_col="XGB_Pred"):
    eval_df = df[[y_col, lin_col, xgb_col]].dropna()
    
    y = eval_df[y_col].values
    p_lin = eval_df[lin_col].values
    p_xgb = eval_df[xgb_col].values
    
    metrics = []
    
    for name, pred in [("Linear Model", p_lin), ("XGBoost Model", p_xgb)]:
        # خطاهای پایه
        rmse = np.sqrt(mean_squared_error(y, pred))
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)
        
        # توزیع و حداکثر خطا
        errors = np.abs(y - pred)
        max_err = np.max(errors)
        p95_err = np.percentile(errors, 95)
        
        # دقت جهت‌یابی (Hit Rate / Directional Accuracy)
        hit_rate = np.mean(np.sign(y) == np.sign(pred)) * 100
        
        # ضریب اطلاعاتی (IC & Rank IC)
        ic, _ = pearsonr(y, pred)
        rank_ic, _ = spearmanr(y, pred)
        
        # بازده فرضی استراتژی جهت‌یابی (Sign Strategy)
        strat_returns = np.sign(pred) * y
        sharpe = (np.mean(strat_returns) / np.std(strat_returns)) * np.sqrt(252) if np.std(strat_returns) != 0 else 0
        cum_ret = np.sum(strat_returns)
        
        metrics.append({
            "Metric": name,
            "RMSE": round(rmse, 5),
            "MAE": round(mae, 5),
            "R2 OOS": round(r2, 4),
            "Hit Rate (%)": round(hit_rate, 2),
            "IC (Pearson)": round(ic, 4),
            "Rank IC": round(rank_ic, 4),
            "Max Error": round(max_err, 4),
            "95th% Error": round(p95_err, 4),
            "Sharpe Ratio": round(sharpe, 2),
            "Cum Return": round(cum_ret, 2)
        })
        
    res_df = pd.DataFrame(metrics).set_index("Metric").T
    
    # محاسبه درصد بهبود XGBoost نسبت به مدل خطی
    # برای خطاها (کاهش بهتر است) و برای سایر متریک‌ها (افزایش بهتر است)
    improvements = []
    for metric in res_df.index:
        lin_val = res_df.loc[metric, "Linear Model"]
        xgb_val = res_df.loc[metric, "XGBoost Model"]
        
        if metric in ["RMSE", "MAE", "Max Error", "95th% Error"]:
            imp = ((lin_val - xgb_val) / np.abs(lin_val)) * 100  # کاهش خطا = درصد مثبت
        else:
            imp = ((xgb_val - lin_val) / np.abs(lin_val)) * 100  # افزایش متریگ = درصد مثبت
            
        improvements.append(round(imp, 2))
        
    res_df["XGB Improvement (%)"] = improvements
    return res_df

# تنظیمات نمایش کامل پانداس
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# محاسبه و پرینت نتایج
comparison_table = evaluate_models(results_df)

print("\n" + "=" * 70)
print("📊 DETAILED OUT-OF-SAMPLE MODEL COMPARISON")
print("=" * 70)
print(comparison_table.to_string())
print("=" * 70 + "\n")