# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Loading the dataset
df = pd.read_csv("/home/ibab/Machine_Learning/Tweets.csv")

# Keeping only the columns which are needed
df = df[['text', 'airline_sentiment']]
df = df.dropna()

# Checking class distribution
print(df['airline_sentiment'].value_counts())

# Converting sentiments into numeric labels
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['airline_sentiment'].map(sentiment_map)

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorizing text
vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

kernels = ['linear', 'rbf', 'poly']

for kernel in kernels:
    print(f"\n........Training........")
    model = SVC(kernel=kernel, C=1, gamma='scale')  # C and gamma can be tuned
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    print(f"{kernel} kernel:")
    print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
