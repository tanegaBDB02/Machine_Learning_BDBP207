import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def loading_data():
    data=pd.read_csv('/home/ibab/Machine_Learning/spam_sms.csv')
    #encoding the label column where ham becoming 0 and spam becoming 1
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['v1'])
    return data, label_encoder

def vector_data(data):
    vectorizer = CountVectorizer()
    X=vectorizer.fit_transform(data['v2']) # storing the feature matrix
    y=data['label'] #storing the target labels
    return X, y, vectorizer

def train_model(X_train, y_train):
    # creating and training the multinomial naive bayes classifier
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    print(f"Accuracy:{accuracy:.2f}")

def main():
    data, label_encoder = loading_data()
    X, y, vectorizer = vector_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = train_model(X_train, y_train)
    evaluation(model, X_test, y_test)

if __name__ == "__main__":
    main()