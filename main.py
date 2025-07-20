import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
import joblib
import os

def build_pipeline(num_attribs: list, cat_attribs: list) :
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    full_pipeline = ColumnTransformer([
        ("num_pipeline", num_pipeline, num_attribs),
        ("cat_pipeline", cat_pipeline, cat_attribs),
    ])

    return full_pipeline

def build_model(X_train, y_train, X_test, y_test, pipeline) :
    models = {
        "Linear_Regression": LinearRegression(),
        "Decision_Tree": DecisionTreeRegressor(),
        "Random_Forest": RandomForestRegressor(),
        "Gradient_Boost": GradientBoostingRegressor(),
        "SVR": SVR()
    }

    rmses = {}

    X_preprocessed = pipeline.fit_transform(X_train)

    for key, value in models.items() :
        value.fit(X_preprocessed, y_train)
        preds = value.predict(pipeline.transform(X_test))
        rmses[key] = root_mean_squared_error(y_test, preds)
    
    min_value = min(rmses.values())
    best_model = ""

    for key, value in rmses.items() :
        if value == min_value :
            best_model = key
            break

    return models[best_model]

def strat_split(data: pd.DataFrame, feature: str, test_size: float) :
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

    for train_idx, test_idx in split.split(data, data[feature]) :
        train_df = data.loc[train_idx].drop(["income_cat"], axis=1).copy()
        test_df = data.loc[test_idx].drop(["income_cat"], axis=1).copy()

    y_train = train_df["median_house_value"].copy()
    X_train = train_df.drop(["median_house_value"], axis=1).copy()
    y_test = test_df["median_house_value"].copy()
    X_test = test_df.drop(["median_house_value"], axis=1).copy()
    
    return X_train, X_test, y_train, y_test

def get_attribs(data: pd.DataFrame) :
    num_attribs = data.drop(["ocean_proximity"], axis=1).columns.to_list()
    cat_attribs = ["ocean_proximity"]

    return num_attribs, cat_attribs

if __name__ == "__main__" :
    df = pd.read_csv("housing.csv")
    df["income_cat"] = pd.cut(
        df["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
        )
    
    X_train, X_test, y_train, y_test = strat_split(df, "income_cat", 0.2)

    num_attribs, cat_attribs = get_attribs(X_train)

    if not os.path.exists("model.pkl") :
        pipeline = build_pipeline(num_attribs, cat_attribs)
        model = build_model(X_train, y_train, X_test, y_test, pipeline)
        joblib.dump(pipeline, "pipeline.pkl")
        joblib.dump(model, "model.pkl")
    else :
        model = joblib.load("model.pkl")
        pipeline = joblib.load("pipeline.pkl")

        print(f"{num_attribs} | {cat_attribs}")

        output_df = pd.DataFrame(zip(model.predict(pipeline.transform(X_test)), y_test), columns=["Predicted", "Actual"])
        print(output_df.head())
    
