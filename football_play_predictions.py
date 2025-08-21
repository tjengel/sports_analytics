import pandas as pd
import numpy as np

from sklearn.preprocessing import PowerTransformer
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


# read in data
df = pd.read_csv("play_data.csv")


# Replace "missing" with null so I can perform operations on the column as numeric
df["GAIN"] = np.where(df["GAIN"] == "MISSING", "", df["GAIN"])
df["GAIN"] = pd.to_numeric(df["GAIN"])

# Some formations aren't used much. I'm consolidating those into a broader category
df["simplified_formation"] = np.where(
    df["FORMATION"].str.contains("2x2"),
    "2x2",
    np.where(
        df["FORMATION"].str.contains("2x3"),
        "2x3",
        np.where(
            df["FORMATION"].str.contains("3x2"),
            "3x2",
            np.where(
                df["FORMATION"].str.contains("1x2"),
                "1x2",
                np.where(
                    df["FORMATION"].str.contains("1x3"),
                    "1x3",
                    np.where(
                        df["FORMATION"].str.contains("3x1"),
                        "3x1",
                        np.where(
                            df["FORMATION"].str.contains("2x1"),
                            "2x1",
                            np.where(
                                df["FORMATION"].str.contains("1x4"),
                                "1x4",
                                np.where(
                                    df["FORMATION"].str.contains("4x1"),
                                    "4x1",
                                    np.where(
                                        df["FORMATION"].str.contains("JUMBO"),
                                        "JUMBO",
                                        "Unknown",
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
)


# Filling null pass results with a "not pass" value
df.PASSRESULT.fillna("not_pass", inplace=True)


# Feature engineering
# Creating features that seem likely to impact how many yards are gained on a play

df["is_redzone"] = np.where(df["LOS"] <= 20, 1, 0)
df["is_backed_up"] = np.where(df["LOS"] >= 90, 1, 0)
df["is_rpo"] = np.where(df.PLAYCALL.str.contains("RPO") == True, 1, 0)
df["is_third_long"] = np.where((df["DOWN"] == 3) & (df["DIST"] >= 7), 1, 0)
df["is_goal_to_go"] = np.where(df["LOS"] == df["DIST"], 1, 0)
df["distance_from_midfield"] = (df["LOS"] - 50) ** 2

# Create interaction of down and dist and log it
df["log_down_*_dist"] = np.log(df["DOWN"] * df["DIST"])

# Creating further features for yard-based offensive and defensive performance

df = df.merge(
    df.groupby(["DEFTEAM", "PLAYTYPE"])["GAIN"]
    .mean()
    .reset_index()
    .rename(columns={"GAIN": "avg_yds_given_up"}),
    on=["DEFTEAM", "PLAYTYPE"],
    how="inner",
)

df = df.merge(
    df.groupby(["FORMATION", "DEFTEAM"])["GAIN"]
    .mean()
    .reset_index()
    .rename(columns={"GAIN": "avg_yds_given_up_formation"}),
    on=["DEFTEAM", "FORMATION"],
    how="left",
)

df = df.merge(
    df.groupby(["DEFTEAM", "DOWN"])["GAIN"]
    .mean()
    .reset_index()
    .rename(columns={"GAIN": "avg_yds_given_up_down"}),
    on=["DEFTEAM", "DOWN"],
    how="left",
)

# RB ypc
df = df.merge(
    df[(df["PLAYTYPE"] == "RUN") & (df["KEY.PLAYER.POSITION"] == "RB")]
    .groupby(["KEY.PLAYER.NUMBER", "PLAYTYPE"])["GAIN"]
    .mean()
    .reset_index()
    .rename(columns={"GAIN": "ypc"}),
    on=["KEY.PLAYER.NUMBER", "PLAYTYPE"],
    how="left",
)

df = df.merge(
    df.groupby(["simplified_formation"])["GAIN"]
    .mean()
    .reset_index()
    .rename(columns={"GAIN": "avg_yds_gained_by_formation"}),
    on=["simplified_formation"],
    how="left",
)

# Subset of columns I will use to train model
training = df[
    [
        "DOWN",
        "DIST",
        "log_down_*_dist",
        "LOS",
        "GAIN",
        "PLAYTYPE",
        "SCOREDIFF",
        "DEFTEAM",
        "simplified_formation",
        "KEY.PLAYER.POSITION",
        "KEY.PLAYER.NUMBER",
        "PASSER",
        "PASSRESULT",
        "is_rpo",
        "is_redzone",
        "is_backed_up",
        "is_third_long",
        "is_goal_to_go",
        "distance_from_midfield",
        "avg_yds_given_up",
        "avg_yds_given_up_formation",
        "avg_yds_given_up_down",
        "ypc",
        "avg_yds_gained_by_formation",
    ]
]

# one-hot encode dummy variables and drop one
training = pd.get_dummies(
    data=training,
    columns=[
        "simplified_formation",
        "PLAYTYPE",
        "DEFTEAM",
        "KEY.PLAYER.POSITION",
        "KEY.PLAYER.NUMBER",
        "PASSER",
        "PASSRESULT",
        "is_rpo",
        "is_redzone",
        "is_backed_up",
        "is_third_long",
        "is_goal_to_go",
    ],
    drop_first=True,
)

# Remove plays to be predicted from data used for model building
complete_df = training.query("GAIN.isna() == False", engine="python")

# Filtering data to plays where the scorediff is between
# -42 and 42. Reason being scores outside this bound could be
# considered "not competitive" and not reflect normal plays
complete_df = complete_df.query("SCOREDIFF >= -42 & SCOREDIFF <= 42")

# Filter to non-null gain
training.query("GAIN.isna() == False", engine="python")


# Transform GAIN since it's positive skewed
pt = PowerTransformer()

pt.fit(complete_df[["GAIN"]])

complete_df["pt_gain"] = pt.transform(complete_df[["GAIN"]])

# Create list of all IVs for model
x_columns = [
    "DOWN",
    "DIST",
    "SCOREDIFF",
    "PLAYTYPE_RUN",
    "is_rpo_1",
    "is_redzone_1",
    "is_backed_up_1",
    "avg_yds_given_up",
    "avg_yds_given_up_formation",
    "avg_yds_given_up_down",
    "avg_yds_gained_by_formation",
    "is_third_long_1",
    "distance_from_midfield",
    "is_goal_to_go_1",
    "ypc",
    "log_down_*_dist",
    "simplified_formation_1x3",
    "simplified_formation_1x4",
    "simplified_formation_2x1",
    "simplified_formation_2x2",
    "simplified_formation_2x3",
    "simplified_formation_3x1",
    "simplified_formation_3x2",
    "simplified_formation_4x1",
    "simplified_formation_JUMBO",
    "PASSER_Mikey Thomas",
    "PASSER_Non-QB",
    "PASSER_Pablo Sanchez",
    "PASSER_Pete Wheeler",
    "PASSRESULT_INCOMPLETE",
    "PASSRESULT_INTERCEPTION",
    "PASSRESULT_SACK",
    "PASSRESULT_SCRAMBLE",
    "PASSRESULT_not_pass",
    "KEY.PLAYER.POSITION_RB",
    "KEY.PLAYER.POSITION_WR1",
    "KEY.PLAYER.POSITION_WR2",
    "KEY.PLAYER.POSITION_WR3",
    "KEY.PLAYER.POSITION_WR4",
    "KEY.PLAYER.NUMBER_3.0",
    "KEY.PLAYER.NUMBER_4.0",
    "KEY.PLAYER.NUMBER_5.0",
    "KEY.PLAYER.NUMBER_6.0",
    "KEY.PLAYER.NUMBER_8.0",
    "KEY.PLAYER.NUMBER_9.0",
    "KEY.PLAYER.NUMBER_10.0",
    "KEY.PLAYER.NUMBER_11.0",
    "KEY.PLAYER.NUMBER_12.0",
    "KEY.PLAYER.NUMBER_13.0",
    "KEY.PLAYER.NUMBER_14.0",
    "KEY.PLAYER.NUMBER_15.0",
    "KEY.PLAYER.NUMBER_17.0",
    "KEY.PLAYER.NUMBER_18.0",
    "KEY.PLAYER.NUMBER_19.0",
    "KEY.PLAYER.NUMBER_24.0",
    "KEY.PLAYER.NUMBER_25.0",
    "KEY.PLAYER.NUMBER_28.0",
    "KEY.PLAYER.NUMBER_29.0",
    "KEY.PLAYER.NUMBER_36.0",
    "KEY.PLAYER.NUMBER_42.0",
    "KEY.PLAYER.NUMBER_52.0",
    "KEY.PLAYER.NUMBER_67.0",
    "KEY.PLAYER.NUMBER_77.0",
    "KEY.PLAYER.NUMBER_80.0",
    "KEY.PLAYER.NUMBER_81.0",
    "KEY.PLAYER.NUMBER_82.0",
    "KEY.PLAYER.NUMBER_83.0",
    "KEY.PLAYER.NUMBER_84.0",
    "KEY.PLAYER.NUMBER_85.0",
    "KEY.PLAYER.NUMBER_86.0",
    "KEY.PLAYER.NUMBER_87.0",
    "KEY.PLAYER.NUMBER_89.0",
    "KEY.PLAYER.NUMBER_93.0",
]

y = complete_df["pt_gain"]

x = complete_df[x_columns]

dmatrix = xgb.DMatrix(data=x, label=y)

params_cv = {
    "objective": "reg:squarederror",
    "max_depth": 1,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.3,
    "colsample_bylevel": 0.8,
    "reg_lambda": 1,
    "eval_metric": "rmse",
    "random_state": 42,
}

reg_cv = xgb.cv(
    dtrain=dmatrix,
    nfold=3,
    params=params_cv,
    num_boost_round=5,
    early_stopping_rounds=10,
    metrics="rmse",
    as_pandas=True,
    seed=123,
)


# Split data into training and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Create param grid that I will perform grid search hyper parameter tuning over
params = {
    "max_depth": [1, 3],
    "learning_rate": [0.001, 0.01, 0.1, 0.3],
    "n_estimators": [100, 200],
    "colsample_bytree": [0.1, 0.3, 0.7],
    "reg_lambda": [1, 10, 100],
    "gamma": [0, 5],
    "alpha": [0, 5],
}

# Fit regressor to training data
reg_h = xgb.XGBRegressor(seed=42)

reg_hyper = GridSearchCV(
    estimator=reg_h,
    param_grid=params,
    scoring="neg_mean_squared_error",
    verbose=1,
    cv=4,
)

reg_hyper.fit(X_train, y_train)


# Printing best fitting model parameters and performance from grid search
print("Best parameters:", reg_hyper.best_params_)
print("Lowest RMSE: ", round(np.sqrt(-reg_hyper.best_score_), 4))


# Fit new model with hyperparameters from best model above
reg_fit = xgb.XGBRegressor(
    colsample_bytree=0.3,
    learning_rate=0.3,
    max_depth=3,
    gamma=0,
    n_estimators=100,
    reg_lambda=100,
    alpha=0,
    seed=42,
)

reg_fit.fit(
    X_train, y_train, verbose=False, eval_set=[(X_train, y_train), (X_test, y_test)]
)


# Print training and test r-squared for the model fit
print(
    "Train r2: {}".format(round(reg_fit.score(X_train, y_train, sample_weight=None), 3))
)
print("Test r2: {}".format(round(reg_fit.score(X_test, y_test, sample_weight=None), 3)))


# Reshape and get mean square error
print(
    mean_squared_error(
        pt.inverse_transform(np.array(y_test).reshape(-1, 1)),
        pt.inverse_transform(np.array(reg_fit.predict(X_test)).reshape(-1, 1)),
        squared=False,
    )
)


# Create df to fit predictions with
predictions = training.query("GAIN.isna() == True", engine="python")

# inverse transform predictions to original scale and save to "gain" column
predictions["GAIN"] = pt.inverse_transform(
    np.array(reg_fit.predict(predictions[x_columns])).reshape(-1, 1)
)


# Join df back to get play id for each prediction
final_df = predictions.merge(
    df["PLAYID"], how="inner", left_index=True, right_index=True
)[["PLAYID", "GAIN"]]
