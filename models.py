import numpy as np
import pandas as pd
import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

def remove_punctuations(text):
    punctuations = string.punctuation + "0123456789"
    res = []
    for _ in text:
        res.append("".join([char for char in _ if char not in punctuations]))
    return res

train = pd.read_csv(r"data/Poem_classification_train_data.csv")
# print(train.shape)
# train.head()
train.dropna(inplace=True)
# train.shape
test = pd.read_csv(r"data/Poem_classification_test_data.csv")
# print(test.shape)
# test.head()

# cv = CountVectorizer(stop_words="english")

# train["Poem"] = train["Poem"].apply(_punctuations)
# test["Poem"] = test["Poem"].apply(_punctuations)
# print(train.head())

X_train = train["Poem"]
y_train = train["Genre"]

X_test =test["Poem"]
y_test = test["Genre"]

# X_train_vect = cv.fit_transform(X_train).toarray()
# X_test_vect = cv.transform(X_test).toarray()

# X_train_df = pd.DataFrame(X_train_vect, columns=cv.get_feature_names_out())
# X_test_df = pd.DataFrame(X_test_vect, columns=cv.get_feature_names_out())
# print(X_train_df.shape)
# X_train_df.head()

clf = make_pipeline(FunctionTransformer(remove_punctuations), CountVectorizer(stop_words="english"), RandomForestClassifier(random_state=42))
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test);

joblib.dump(clf, "rf_model.sav")

def model():
    return clf