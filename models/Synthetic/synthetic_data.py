# utils/generate_synthetic_data.py
import pandas as pd
import numpy as np


def generate_synthetic_data(file_name="synthetic_testbench_data.csv"):
    np.random.seed(42)
    time_steps = np.linspace(0, 100, num=500)
    speed_profile = np.concatenate([
        np.linspace(0, 2500, num=250),
        np.linspace(2500, 0, num=250)
    ])

    flow_rate = np.random.uniform(10, 100, len(time_steps))
    pressure2 = 0.07 * speed_profile + np.random.normal(0, 50, len(speed_profile))
    pressure1 = 0.09 * speed_profile + np.random.normal(0, 50, len(speed_profile))

    synthetic_data = pd.DataFrame({
        "Zeit [s]": time_steps,
        "A: flow rs400 [l/min]": flow_rate,
        "B: Druck2 [bar]": pressure2,
        "C: Druck1 [bar]": pressure1,
        "I: drehzahl [U/min]": speed_profile
    })
    synthetic_data.to_csv(file_name, index=False)
    print(f"Synthetic data saved to {file_name}")