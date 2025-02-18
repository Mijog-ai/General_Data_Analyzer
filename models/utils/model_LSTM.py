'''
import xgboost as xgb
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_model(df, max_depth=10, n_estimators=250, learning_rate=0.2):
    """
    Train an XGBoost model to predict Pressure1 and Pressure2 based on Speed and Flow.

    Parameters:
        df (pd.DataFrame): The input data frame.
        max_depth (int): Maximum depth of the trees.
        n_estimators (int): Number of boosting rounds.
        learning_rate (float): Step size shrinkage.

    Returns:
        xgb.Booster: Trained XGBoost model.
    """
    X = df[["Speed", "Flow"]].values
    y = df[["Pressure1", "Pressure2"]].values
    print(y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=52)

    # Convert to XGBoost DMatrix format
    train_data = xgb.DMatrix(X_train, label=y_train)
    test_data = xgb.DMatrix(X_test, label=y_test)

    # Define XGBoost parameters
    params = {
        "objective": "reg:squarederror",  # Regression task
        "max_depth": max_depth,
        # "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "eval_metric": "rmse"  # Root Mean Squared Error
    }

    # Train the model
    model = xgb.train(params, train_data, num_boost_round=n_estimators, evals=[(test_data, "test")])

    # Predict and evaluate
    y_pred = model.predict(test_data)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return model


def save_model(model, path):
    """
    Save the XGBoost model to a file.
    """
    model.save_model(path)


def load_model(path):
    """
    Load the XGBoost model from a file.
    """
    model = xgb.Booster()
    model.load_model(path)
    return model
'''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import qmc  # For Latin Hypercube Sampling (LHS)


def lhs_sampling(data, n_samples):
    """
    Perform Latin Hypercube Sampling (LHS) on the dataset.

    Parameters:
        data (pd.DataFrame): The full dataset.
        n_samples (int): Number of samples to select.

    Returns:
        pd.DataFrame: Sampled dataset.
    """
    sampler = qmc.LatinHypercube(d=len(data.columns))

    # Generate LHS samples and scale them to the row indices range
    sample_indices = qmc.scale(sampler.random(n=n_samples), 0, len(data) - 1)

    # Convert to integers for indexing
    sample_indices = np.floor(sample_indices[:, 0]).astype(int)  # Select first column, since indices are 1D

    return data.iloc[sample_indices]


def train_active_learning_model(df, n_samples=50, n_estimators=100, max_depth=5):
    """
    Train a RandomForest model using Active Learning and LHS.

    Parameters:
        df (pd.DataFrame): The input data.
        n_samples (int): Number of initial LHS samples.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): The maximum depth of the trees.

    Returns:
        RandomForestRegressor: Trained model.
    """
    # Feature and target selection
    X = df[["Speed", "Flow"]].values
    y = df[["Pressure1", "Pressure2"]].values

    # Perform LHS for selecting initial training samples
    sampled_df = lhs_sampling(df, n_samples)

    X_sampled = sampled_df[["Speed", "Flow"]].values
    y_sampled = sampled_df[["Pressure1", "Pressure2"]].values

    # Split the sampled data
    X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)

    # Train a RandomForestRegressor
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return model

def save_model(model, path):
    """
    Save the trained model.
    """
    import joblib
    joblib.dump(model, path)

def load_model(path):
    """
    Load the saved model.
    """
    import joblib
    return joblib.load(path)
