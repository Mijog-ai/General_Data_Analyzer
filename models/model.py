'''
import models.utils.data_utils as dp
import models.utils.model_LSTM as mdl  # Change from model_LSTM to model_XGBoost
import models.utils.predicts as pred
import models.utils.upload as upl

if __name__ == "__main__":
    # Step 1: Train the XGBoost model
    data_file = "../General_Data_analyzer/v60n 54_1ccm_-000012_250127-134748.csv"  # Replace with actual testbench data file
    df, scaler = dp.load_and_preprocess_data(data_file)
    model = mdl.train_model(df)  # XGBoost training

    # Step 2: Save and Deploy Model
    hf_repo_id = "InlineHydraulik/Test"  # Replace with actual Hugging Face repo
    mdl.save_model(model, "v60n_xgboost_model.json")  # Save as JSON for XGBoost
    dp.save_scaler(scaler, "scaler.pkl")

    # Upload to Hugging Face
    upl.upload_to_huggingface("v60n_xgboost_model.json", hf_repo_id, "v60n_xgboost_model.json")
    upl.upload_to_huggingface("scaler.pkl", hf_repo_id, "scaler.pkl")

    # Step 3: Load and Predict on new pump data
    new_pump_file = "../General_Data_analyzer/v60n 54_1ccm_-000012_250127-134748.csv"  # Replace with actual new test data file
    model = mdl.load_model("v60n_xgboost_model.json")  # Load XGBoost model
    scaler = dp.load_scaler("scaler.pkl")
    predictions_df = pred.predict_new_pump(new_pump_file, model, scaler)

    # Step 4: Display results
    print(predictions_df.head())  # Print predictions instead of displaying with tools
'''

import models.utils.data_utils as dp
import models.utils.model_LSTM as mdl  # Updated import
import models.utils.predicts as pred
import models.utils.upload as upl

if __name__ == "__main__":
    # Load and preprocess data
    data_file = "../General_Data_analyzer/v60n 54_1ccm_-000012_250127-134748.csv"
    df, scaler = dp.load_and_preprocess_data(data_file)

    # Train the Active Learning Model
    model = mdl.train_active_learning_model(df, n_samples=50)  # Using LHS sampling

    # Save and upload the model
    hf_repo_id = "InlineHydraulik/Test"
    mdl.save_model(model, "v60n_active_learning_model.pkl")
    dp.save_scaler(scaler, "scaler.pkl")

    upl.upload_to_huggingface("v60n_active_learning_model.pkl", hf_repo_id, "v60n_active_learning_model.pkl")
    upl.upload_to_huggingface("scaler.pkl", hf_repo_id, "scaler.pkl")

    # Load and predict with the trained model
    new_pump_file = "../General_Data_analyzer/v60n 54_1ccm_-000012_250127-134748.csv"
    model = mdl.load_model("v60n_active_learning_model.pkl")
    scaler = dp.load_scaler("scaler.pkl")
    predictions_df = pred.predict_new_pump(new_pump_file, model, scaler)

    # Display predictions
    print(predictions_df.head())
