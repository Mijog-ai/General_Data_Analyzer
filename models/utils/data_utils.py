# utils/data_processing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_and_preprocess_data(file_path):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    # Use sep=";" to correctly read the German CSV format
    df = pd.read_csv(file_path, sep=";")

    # Rename columns to match expected names
    df.rename(columns={
        "Zeit [s]": "Time",
        "A: flow rs400 [l/min]": "Flow",
        "B: Druck2 [bar]": "Pressure2",
        "C: Druck1 [bar]": "Pressure1",
        "I: drehzahl [U/min]": "Speed"
    }, inplace=True)

    # Convert German decimal format (comma to dot)
    for col in ["Speed", "Flow", "Pressure1", "Pressure2"]:
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    df[["Speed", "Flow", "Pressure1", "Pressure2"]] = scaler.fit_transform(df[["Speed", "Flow", "Pressure1", "Pressure2"]])

    return df, scaler


def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)