import pandas as pd

file_path = "/home/ibab/Machine_Learning/simulated_data_multiple_linear_regression_for_ML.csv"
data = pd.read_csv(file_path)

def partition(data,threshold=80):
    left_partition=data[data["BP"]<= threshold]
    right_partition=data[data["BP"]> threshold]
    return left_partition,right_partition

def main():
    thresholds = [78, 80, 82]
    for t in thresholds:
        left, right = partition(data, t)
        print(f"Threshold t={t}: Lower Partition={len(left)}, Upper Partition={len(right)}")
        print(f"Sample from Lower Partition:\n{left}\n")
        print(f"Sample from Upper Partition:\n{right}\n")

if __name__ == "__main__":
    main()

