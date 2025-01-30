# main.py
import models.utils.data_utils as dp
import models.utils.model_LSTM as mdl
import models.utils.predicts as pred
import models.utils.upload as upl
import ace_tools as tools

if __name__ == "__main__":
    # Step 1: Train the model
    data_file = "testbench_data.csv"  # Replace with actual testbench data file
    df, scaler = dp.load_and_preprocess_data(data_file)
    model = mdl.train_model(df)

    # Step 2: Save and Deploy Model
    hf_repo_id = "your-hf-username/v60n_pump_models"  # Replace with actual Hugging Face repo
    mdl.save_model(model, "v60n_lstm_model.pth")
    dp.save_scaler(scaler, "scaler.pkl")
    upl.upload_to_huggingface("v60n_lstm_model.pth", hf_repo_id, "v60n_lstm_model.pth")
    upl.upload_to_huggingface("scaler.pkl", hf_repo_id, "scaler.pkl")

    # Step 3: Load and Predict on new pump data
    new_pump_file = "new_pump_test.csv"  # Replace with actual new test data file
    model = mdl.load_model("v60n_lstm_model.pth")
    scaler = dp.load_scaler("scaler.pkl")
    predictions_df = pred.predict_new_pump(new_pump_file, model, scaler)

    # Step 4: Display results
    tools.display_dataframe_to_user(name="New Pump Predictions", dataframe=predictions_df)
