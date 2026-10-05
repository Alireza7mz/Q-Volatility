


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# فرض بر این است که داده‌های شما درون متغیری به نام df ذخیره شده‌اند
df = pd.read_csv("~/Documents/src/xgb_final_predictions.csv")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# فرض بر این است که نام ستون‌های df مطابق تصویر باشد: Actual, XGB_Pred, Linear_Pred
# df = pd.read_csv('your_file.csv')

# ۱. محاسبه خطاها و معیارهای ارزیابی
df['error_linear'] = df['Linear_Pred'] - df['Actual']
df['error_xgb'] = df['XGB_Pred'] - df['Actual']

r2_lin = r2_score(df['Actual'], df['Linear_Pred'])
mae_lin = mean_absolute_error(df['Actual'], df['Linear_Pred'])
rmse_lin = np.sqrt(mean_squared_error(df['Actual'], df['Linear_Pred']))

r2_xgb = r2_score(df['Actual'], df['XGB_Pred'])
mae_xgb = mean_absolute_error(df['Actual'], df['XGB_Pred'])
rmse_xgb = np.sqrt(mean_squared_error(df['Actual'], df['XGB_Pred']))

# ---------------------------------------------------------
# نمودار اول: مقایسه همبستگی Actual vs Predicted (دو پنل)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True, sharex=True)

min_val = min(
    df['Actual'].min(), df['Linear_Pred'].min(), df['XGB_Pred'].min()
)
max_val = max(
    df['Actual'].max(), df['Linear_Pred'].max(), df['XGB_Pred'].max()
)

# پنل ۱: مدل خطی
sns.scatterplot(
    ax=axes[0],
    x=df['Actual'],
    y=df['Linear_Pred'],
    alpha=0.4,
    color='#1f77b4',
    s=30,
)
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
axes[0].set_title(
    f'Linear Model Baseline\n$R^2 = {r2_lin:.4f}$ | MAE = {mae_lin:.4f}',
    fontsize=11,
)
axes[0].set_xlabel('مقدار واقعی (Actual)', fontsize=10)
axes[0].set_ylabel('مقدار پیش‌بینی (Predicted)', fontsize=10)
axes[0].grid(True, linestyle=':', alpha=0.6)

# پنل ۲: مدل XGBoost
sns.scatterplot(
    ax=axes[1],
    x=df['Actual'],
    y=df['XGB_Pred'],
    alpha=0.4,
    color='#2ca02c',
    s=30,
)
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
axes[1].set_title(
    f'XGBoost Final Model\n$R^2 = {r2_xgb:.4f}$ | MAE = {mae_xgb:.4f}',
    fontsize=11,
)
axes[1].set_xlabel('مقدار واقعی (Actual)', fontsize=10)
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.suptitle(
    'ارزیابی و مقایسه مدل خطی با XGBoost در پیش‌بینی $\ln(RS)$', fontsize=13
)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# نمودار دوم: هم‌پوشانی توزیع خطای دو مدل (Residuals Overlay)
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))

sns.kdeplot(
    df['error_linear'],
    color='#1f77b4',
    linewidth=2,
    label=f'Linear Error (MAE: {mae_lin:.4f})',
    fill=True,
    alpha=0.2,
)
sns.kdeplot(
    df['error_xgb'],
    color='#2ca02c',
    linewidth=2,
    label=f'XGBoost Error (MAE: {mae_xgb:.4f})',
    fill=True,
    alpha=0.3,
)

plt.axvline(0, color='black', linestyle='--', linewidth=1.2, label='Zero Error')

plt.title('مقایسه چگالی خطای مدل خطی و XGBoost', fontsize=12)
plt.xlabel('میزان خطا (Predicted - Actual)', fontsize=11)
plt.ylabel('چگالی (Density)', fontsize=11)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()

plt.show()