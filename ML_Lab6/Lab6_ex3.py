import pandas as pd
import numpy as np

def standardize_data():
    file_path = "/home/ibab/Machine_Learning/breast_cancer_data.csv"
    data = pd.read_csv(file_path)

    data = data.drop(columns=['id', "Unnamed: 32"])
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    for col in data.columns:
        if col != 'diagnosis':
            mean_val = np.mean(data[col])
            std_val = np.std(data[col])
            standardized_values = (data[col] - mean_val) / std_val

            standardized_mean = np.mean(standardized_values)
            standardized_std = np.std(standardized_values)

            print(f"Standardized values for '{col}' (first 5 values):")
            print(standardized_values[:5])
            print()

            print(f"After standardizing '{col}':")
            print(f"  Mean: {standardized_mean}")
            print(f"  Standard Deviation: {standardized_std}")
            print()


def main():
    standardize_data()

if __name__ == "__main__":
    main()
