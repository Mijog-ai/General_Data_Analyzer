'''
import pandas as pd
import numpy as np
import xgboost as xgb

def predict_new_pump(file_path, model, scaler):
    # Load test data with safe error handling
    new_data = pd.read_csv(file_path, sep=';', on_bad_lines="skip")

    # Print detected columns for debugging
    print("CSV Columns Found:", new_data.columns)

    # Rename columns to match training data
    new_data.rename(columns={
        "Zeit [s]": "Time",
        "A: flow rs400 [l/min]": "Flow",
        "B: Druck2 [bar]": "Pressure2",
        "C: Druck1 [bar]": "Pressure1",
        "I: drehzahl [U/min]": "Speed"
    }, inplace=True)

    # Ensure all required columns exist before transformation
    required_features = ["Time", "Speed", "Flow", "Pressure1", "Pressure2"]
    missing_features = [col for col in required_features if col not in new_data.columns]

    if missing_features:
        raise ValueError(f"Missing features in test data: {missing_features}")

    # Convert decimal commas to dots
    for col in ["Speed", "Flow", "Pressure1", "Pressure2"]:
        new_data[col] = new_data[col].astype(str).str.replace(",", ".").astype(float)

    # Apply transformation using MinMaxScaler
    transformed_features = scaler.transform(new_data[["Speed", "Flow", "Pressure1", "Pressure2"]])

    # Create a DataFrame from the transformed values
    new_data_scaled = pd.DataFrame(transformed_features, columns=["Speed", "Flow", "Pressure1", "Pressure2"])

    # Ensure "Time" is added correctly
    new_data_scaled["Time"] = new_data["Time"]

    # Convert input data to XGBoost DMatrix format
    input_data = xgb.DMatrix(new_data_scaled[["Speed", "Flow"]].values)

    # Get predictions using XGBoost
    predictions = model.predict(input_data)

    # Convert predictions to a NumPy array
    predictions = np.array(predictions)

    # Ensure correct shape (2D)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 2)

    # **Inverse transform predictions to original scale**
    # The scaler was fitted on ["Speed", "Flow", "Pressure1", "Pressure2"], so we must create a placeholder array
    placeholder_array = np.zeros((predictions.shape[0], 2))  # Placeholder for Speed & Flow
    predictions_full = np.hstack((placeholder_array, predictions))  # Add placeholders

    # Apply inverse transform
    predictions_original_scale = scaler.inverse_transform(predictions_full)[:, 2:]  # Extract Pressure1 & Pressure2

    # Store correctly scaled predictions in DataFrame
    new_data["Predicted_Pressure1"] = predictions_original_scale[:, 0]
    new_data["Predicted_Pressure2"] = predictions_original_scale[:, 1]

    return new_data

# utils/upload.py
from huggingface_hub import HfApi

def upload_to_huggingface(file_path, repo_id, filename):
    api = HfApi()
    api.upload_file(path_or_fileobj=file_path, path_in_repo=filename, repo_id=repo_id)
    print(f"Uploaded {filename} to {repo_id}")
'''
import pandas as pd
import numpy as np
import joblib

def predict_new_pump(file_path, model, scaler):
    new_data = pd.read_csv(file_path, sep=';', on_bad_lines="skip")

    # Rename columns
    new_data.rename(columns={
        "Zeit [s]": "Time",
        "A: flow rs400 [l/min]": "Flow",
        "B: Druck2 [bar]": "Pressure2",
        "C: Druck1 [bar]": "Pressure1",
        "I: drehzahl [U/min]": "Speed"
    }, inplace=True)

    required_features = ["Speed", "Flow"]
    if any(col not in new_data.columns for col in required_features):
        raise ValueError(f"Missing required features: {required_features}")

    # Convert comma-decimals
    for col in ["Speed", "Flow"]:
        new_data[col] = new_data[col].astype(str).str.replace(",", ".").astype(float)

    # Normalize data
    # Ensure all required columns exist for transformation
    new_data["Pressure1"] = 0  # Placeholder
    new_data["Pressure2"] = 0  # Placeholder

    # Apply transformation using the scaler fitted on ["Speed", "Flow", "Pressure1", "Pressure2"]
    transformed_features = scaler.transform(new_data[["Speed", "Flow", "Pressure1", "Pressure2"]])

    # Remove placeholders after transformation
    transformed_features = transformed_features[:, :2]  # Keep only Speed & Flow

    new_data_scaled = pd.DataFrame(transformed_features, columns=["Speed", "Flow"])

    # Predict using RandomForest
    predictions = model.predict(new_data_scaled)

    # Convert to original scale
    placeholder_array = np.zeros((predictions.shape[0], 2))  # Placeholder
    predictions_full = np.hstack((placeholder_array, predictions))
    predictions_original_scale = scaler.inverse_transform(predictions_full)[:, 2:]

    # Store predictions in DataFrame
    new_data["Predicted_Pressure1"] = predictions_original_scale[:, 0]
    new_data["Predicted_Pressure2"] = predictions_original_scale[:, 1]

    return new_data
