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
        "learning_rate": learning_rate,
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
