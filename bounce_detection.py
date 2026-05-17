import pandas as pd
import numpy as np
import catboost as ctb

# NUM_FEATURE_FRAMES = 3
# eps = 1e-15

# df must contain one row per frame with columns:
#   x-coordinate, y-coordinate
df = pd.read_csv("output.csv")

for i in range(1, NUM_FEATURE_FRAMES):
    df[f'x_lag_{i}'] = df['x-coordinate'].shift(i)
    df[f'x_lag_inv_{i}'] = df['x-coordinate'].shift(-i)
    df[f'y_lag_{i}'] = df['y-coordinate'].shift(i)
    df[f'y_lag_inv_{i}'] = df['y-coordinate'].shift(-i)

    df[f'x_diff_{i}'] = abs(df[f'x_lag_{i}'] - df['x-coordinate'])
    df[f'y_diff_{i}'] = df[f'y_lag_{i}'] - df['y-coordinate']
    df[f'x_diff_inv_{i}'] = abs(df[f'x_lag_inv_{i}'] - df['x-coordinate'])
    df[f'y_diff_inv_{i}'] = df[f'y_lag_inv_{i}'] - df['y-coordinate']

    df[f'x_div_{i}'] = abs(df[f'x_diff_{i}'] / (df[f'x_diff_inv_{i}'] + eps))
    df[f'y_div_{i}'] = df[f'y_diff_{i}'] / (df[f'y_diff_inv_{i}'] + eps)

colnames_x = [f'x_diff_{i}' for i in range(1, NUM_FEATURE_FRAMES)] + \
             [f'x_diff_inv_{i}' for i in range(1, NUM_FEATURE_FRAMES)] + \
             [f'x_div_{i}' for i in range(1, NUM_FEATURE_FRAMES)]

colnames_y = [f'y_diff_{i}' for i in range(1, NUM_FEATURE_FRAMES)] + \
             [f'y_diff_inv_{i}' for i in range(1, NUM_FEATURE_FRAMES)] + \
             [f'y_div_{i}' for i in range(1, NUM_FEATURE_FRAMES)]

feature_cols = colnames_x + colnames_y
# df = df.dropna(subset=feature_cols)

# model = ctb.CatBoostRegressor()
# model.load_model("weights\\bounce_model.cbm")

scores = model.predict(df[feature_cols])
df["scores"] = scores
df["bounce_pred"] = (scores > 0.5).astype(int)
df[['x-coordinate', 'y-coordinate', "scores", "bounce_pred"]].to_csv("output_bounce.csv", index=False)