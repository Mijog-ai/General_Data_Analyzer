# main.py
import  models.Synthetic.synthetic_data  as synth
import  models.Synthetic.Test_model as test

if __name__ == "__main__":
    # Generate synthetic data
    synth.generate_synthetic_data("synthetic_train_data.csv")
    synth.generate_synthetic_data("synthetic_test_data.csv")

    # Train, upload, and test the model
    hf_repo_id = "InlineHydraulik/Test"
    test.test_and_upload_model("synthetic_train_data.csv", "synthetic_test_data.csv", hf_repo_id)