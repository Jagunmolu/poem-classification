import numpy as np
import pandas as pd
# import joblib
import matplotlib.pyplot as plt
# import seaborn as sns
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# def remove_punctuations(text):
#     punctuations = string.punctuation + "0123456789"
#     res = []
#     for _ in text:
#         res.append("".join([char for char in _ if char not in punctuations]))
#     return res

# train = pd.read_csv(r"data/Poem_classification_train_data.csv")
# print(train.shape)
# train.head()
# train.dropna(inplace=True)
# train.shape
# test = pd.read_csv(r"data/Poem_classification_test_data.csv")
# print(test.shape)
# test.head()

# cv = CountVectorizer(stop_words="english")

# train["Poem"] = train["Poem"].apply(_punctuations)
# test["Poem"] = test["Poem"].apply(_punctuations)
# print(train.head())

# X_train = train["Poem"]
# y_train = train["Genre"]

# X_test =test["Poem"]
# y_test = test["Genre"]

# X_train_vect = cv.fit_transform(X_train).toarray()
# X_test_vect = cv.transform(X_test).toarray()

# X_train_df = pd.DataFrame(X_train_vect, columns=cv.get_feature_names_out())
# X_test_df = pd.DataFrame(X_test_vect, columns=cv.get_feature_names_out())
# print(X_train_df.shape)
# X_train_df.head()

# clf = make_pipeline(FunctionTransformer(remove_punctuations), CountVectorizer(stop_words="english"), RandomForestClassifier(random_state=42))
# clf.fit(X_train, y_train)
# print(classification_report(y_test, clf.predict(X_test)))
# cm = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)

# cm = confusion_matrix(y_test, clf.predict(X_test))
# display = ConfusionMatrixDisplay(cm).plot()

# joblib.dump(clf, "rf_model.sav")

# def model():
#     return clf

# def confusion_matrix():
    # print(type(cm))
    # return cm
    # return display

class Models:
    def __init__(self, model):
        self.punctuations = string.punctuation + "0123456789"
        self.clf = None
        self.train = pd.read_csv(r"data/Poem_classification_train_data.csv")
        self.train.dropna(inplace=True)
        self.test = pd.read_csv(r"data/Poem_classification_test_data.csv")
        self.X_train = self.train["Poem"]
        self.y_train = self.train["Genre"]
        self.X_test = self.test["Poem"]
        self.y_test = self.test["Genre"]
        self.pred = None
        self.model = model

    # def feature_engineering(self):
    #     X_train = self.train["Poem"]
    #     y_train = self.train["Genre"]
        
    #     X_test = self.test["Poem"]
    #     y_test = self.test["Genre"]
    #     return (X_train, y_train), (X_test, y_test)
        
    def model_init(self):
        if self.model == "random forest":
            self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), RandomForestClassifier(random_state=42))
        # elif self.model == "svm":
        #     self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), MinMaxScaler(feature_range=(-1, 1)), SVC(random_state=42))
        elif self.model == "svm":
            self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), SVC(random_state=42))
        elif self.model == "xgboost":
            self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), XGBClassifier())
        elif self.model == "decision tree":
            self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), DecisionTreeClassifier(random_state=42))
        elif self.model == "naive bayes multinomial":
            self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), MultinomialNB())
        elif self.model == "naive bayes complement":
            self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), ComplementNB())
        elif self.model == "knn":
            self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), KNeighborsClassifier())
        self.clf.fit(self.X_train, self.y_train)
        print(f"{type(self.clf) = }")

    def metrics(self):
        self.pred = self.clf.predict(self.X_test)
        class_report = classification_report(self.y_test, self.pred)
        return class_report

    # def train_model(self):
    #     self.clf.fit(self.X_train, self.y_train)
        # return 

    def remove_punctuations(self, series):
        res = []
        for _ in series:
            res.append("".join([char for char in _ if char not in self.punctuations]))
        return res

    def save_fig(self):
        cm = confusion_matrix(self.y_test, self.pred)
        # print(type(cm))
        display = ConfusionMatrixDisplay(cm, display_labels=self.clf.classes_).plot()
        # display = ConfusionMatrixDisplay(cm).plot()
        # display = ConfusionMatrixDisplay.from_predictions(self.y_test, self.pred).plot()
        # print(type(display))
        plt.savefig("temp.png")
        # plt.show()

    def final_result(self):
        self.model_init()
        # self.train_model()
        self.metrics()
        self.save_fig()