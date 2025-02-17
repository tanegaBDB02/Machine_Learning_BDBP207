import pandas as pd

def normalize_data():
    file_path = "/home/ibab/Machine_Learning/breast_cancer_data.csv"
    data = pd.read_csv(file_path)

    data = data.drop(columns=['id', "Unnamed: 32"])
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    for col in data.columns:
        min_val = data[col].min()
        max_val = data[col].max()
        normalized_values = (data[col] - min_val) / (max_val - min_val)

        print(f"Normalized values for '{col}' (first 5 values):")
        print(normalized_values[:5].tolist())
        print()


def main():
    normalize_data()

if __name__ == "__main__":
    main()


