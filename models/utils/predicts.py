import torch
import pandas as pd
import numpy as np

def predict_new_pump(file_path, model, scaler):
    # Load test data
    new_data = pd.read_csv(file_path)

    # Rename columns to match training data
    new_data.rename(columns={
        "Zeit [s]": "Time",
        "A: flow rs400 [l/min]": "Flow",
        "B: Druck2 [bar]": "Pressure2",
        "C: Druck1 [bar]": "Pressure1",
        "I: drehzahl [U/min]": "Speed"
    }, inplace=True)

    # Ensure all required columns exist before transformation
    required_features = ["Time", "Speed", "Flow","Pressure1", "Pressure2"]
    missing_features = [col for col in required_features if col not in new_data.columns]

    if missing_features:
        raise ValueError(f"Missing features in test data: {missing_features}")

    # Apply transformation using all features to match training
    transformed_features = scaler.transform(new_data[["Speed", "Flow" ,"Pressure1", "Pressure2"]])

    # Create a DataFrame from the transformed values
    new_data_scaled = pd.DataFrame(transformed_features, columns=["Speed", "Flow" ,"Pressure1", "Pressure2"])

    # Ensure "Time" is added correctly
    new_data_scaled["Time"] = new_data["Time"]

    # Prepare input for the model (reshape to batch format: batch_size=1, seq_len=500, feature_dim=2)
    input_data = torch.tensor(new_data_scaled[[ "Speed", "Flow"]].values, dtype=torch.float32).unsqueeze(0)

    # Debugging: Print input shape before passing to the model
    print(f"Input shape to LSTM: {input_data.shape}")  # Expected: (1, 500, 2)

    # Get predictions for all time steps
    with torch.no_grad():
        predictions = model(input_data)  # Should output shape (1, 500, 2)

    # Debugging: Print prediction shape
    print(f"Raw predictions shape: {predictions.shape}")  # Expected: (1, 500, 2)

    # Remove batch dimension and reshape
    predictions = predictions.squeeze(0).numpy()  # Expected shape (500, 2)

    # Debugging: Print final prediction shape
    print(f"Final predictions shape: {predictions.shape}")  # Expected: (500, 2)

    # Store predictions in DataFrame
    new_data["Predicted_Pressure1"] = predictions[:, 0]
    new_data["Predicted_Pressure2"] = predictions[:, 1]

    return new_data


# utils/upload.py
from huggingface_hub import HfApi

def upload_to_huggingface(file_path, repo_id, filename):
    api = HfApi()
    api.upload_file(path_or_fileobj=file_path, path_in_repo=filename, repo_id=repo_id)
    print(f"Uploaded {filename} to {repo_id}")
