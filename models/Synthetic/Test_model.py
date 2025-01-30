# utils/test_model.py
import torch
import pandas as pd
import models.utils.data_utils as dp
import models.utils.model_LSTM as mdl
import models.utils.upload as upl
import models.utils.predicts as pred


def test_and_upload_model(train_file, test_file, hf_repo_id):
    # Load and preprocess training data
    df, scaler = dp.load_and_preprocess_data(train_file)
    model = mdl.train_model(df)
    mdl.save_model(model, "v60n_lstm_model.pth")
    dp.save_scaler(scaler, "scaler.pkl")

    # Upload trained model to Hugging Face
    upl.upload_to_huggingface("v60n_lstm_model.pth", hf_repo_id, "v60n_lstm_model.pth")
    upl.upload_to_huggingface("scaler.pkl", hf_repo_id, "scaler.pkl")

    # Load trained model and scaler
    model = mdl.load_model("v60n_lstm_model.pth")
    scaler = dp.load_scaler("scaler.pkl")

    # Predict on new test data
    predictions_df = pred.predict_new_pump(test_file, model, scaler)
    predictions_df.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")


