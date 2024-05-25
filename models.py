import pandas as pd
import matplotlib.pyplot as plt
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


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

        
    def model_init(self):
        if self.model == "random forest":
            self.clf = make_pipeline(FunctionTransformer(self.remove_punctuations), CountVectorizer(stop_words="english"), RandomForestClassifier(random_state=42))
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
        classes = self.clf.classes_
        class_report = pd.DataFrame(classification_report(self.y_test, self.pred, target_names=classes, output_dict=True)).transpose()
        return class_report

    def remove_punctuations(self, series):
        res = []
        for _ in series:
            res.append("".join([char for char in _ if char not in self.punctuations]))
        return res

    def save_fig(self):
        cm = confusion_matrix(self.y_test, self.pred)
        display = ConfusionMatrixDisplay(cm, display_labels=self.clf.classes_).plot()
        plt.savefig("temp.png")

    def final_result(self):
        self.model_init()
        self.metrics()
        self.save_fig()