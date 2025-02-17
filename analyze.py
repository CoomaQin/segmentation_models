import pandas as pd
import os
import numpy as np

input_dir = "lightning_logs/BAS_ablation_study"
scenarios = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
col_name = "valid_dataset_area_mf1"
vals = []
for scenario_dir in scenarios:
    csv_path = os.path.join(input_dir, scenario_dir, "metrics.csv")
    df = pd.read_csv(csv_path)
    vals.append(float(df[col_name].iloc[-1]) if float(df[col_name].iloc[-1]) > 0 else float(df[col_name].iloc[-2]))
print(np.std(vals))