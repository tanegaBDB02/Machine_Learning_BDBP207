from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    file_path = "simulated_data_multiple_linear_regression_for_ML.csv"
    data = pd.read_csv(file_path)
    print("Dataset preview:")
    print(data.head())

    X = data.drop(columns=['disease_score', 'disease_score_fluct'])
    #y = data['disease_score_fluct']
    y = data['disease_score']

    return X, y

def EDA(data):
    data.hist(figsize=(12, 10), bins=30, edgecolor="black")
    plt.subplots_adjust(hspace=0.7, wspace=0.4)
    plt.show()

    features_of_interest = ["age", "BMI", "BP", "blood_sugar", "Gender"]
    print(data[features_of_interest].describe())


def box_plot(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
    # boxplot without scaling
    color = {"whiskers": "black", "medians": "black", "caps": "black"}
    plt.figure(figsize=(12, 10))
    data.boxplot(vert=False, color=color, patch_artist=True)
    plt.axvline(x=0, ymin=-1, ymax=1, color="black", linestyle="--")
    _ = plt.title("Boxplot of Features in Dataset before scaling")
    plt.show()
    # boxplot with scaling
    color = {"whiskers": "black", "medians": "black", "caps": "black"}
    plt.figure(figsize=(12, 10))
    scaled_df.boxplot(vert=False, color=color, patch_artist=True)
    plt.axvline(x=0, ymin=-1, ymax=1, color="black", linestyle="--")
    _ = plt.title("Boxplot of Features in Dataset after scaling")
    plt.show()

# def scatter_plot(data):
#     features = ["age", "BMI", "BP", "blood_sugar"]
#     target = "disease_score"
#
#     for feature in features:
#         plt.figure(figsize=(8, 6))
#         plt.scatter(data[feature], data[target], alpha=0.7, edgecolors='k')
#         plt.title(f'Scatter Plot: {feature} vs {target}')
#         plt.xlabel(feature)
#         plt.ylabel(target)
#         plt.grid(True, linestyle="--", alpha=0.5)
#         plt.show()


def scatter_plot(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
    plt.title('Actual vs Predicted Disease Score')
    plt.xlabel('Actual Disease Score')
    plt.ylabel('Predicted Disease Score')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def main():
    X,y=load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)

    ##scale the data
    scaler = StandardScaler()
    scaler= scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #train a model
    print("---Training under progress---")
    print ("N =%d" %(len(X)))
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2= r2_score(y_test, y_pred)
    print ("R2 =", r2)

    data=pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    EDA(data)
    box_plot(data)
    scatter_plot(y_test, y_pred)


if __name__ == '__main__':
    main()