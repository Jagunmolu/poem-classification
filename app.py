import streamlit as st
# import pandas as pd
# import numpy as np
# from prediction import predict
# from models import model, confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from models import Models

st.title("Classifying Poems")
# st.markdown("Toy model to play to classify iris flowers into setosa, versicolor, virginica")

# st.header("Plant Features")
# col1, col2 = st.columns(2)
# with col1:
#     st.text("Sepal characteristics")
#     sepal_l = st.slider("Sepal lenght (cm)", 1.0, 8.0, 0.5)
#     sepal_w = st.slider("Sepal width (cm)", 2.0, 4.4, 0.5)
# with col2:
#     st.text("Pepal characteristics")
#     petal_l = st.slider("Petal lenght (cm)", 1.0, 7.0, 0.5)
#     petal_w = st.slider("Petal width (cm)", 0.1, 2.5, 0.5)

text = st.text_input("Enter the text to classify.")
# text = st.text_area("Enter the text to classify.")
choice = st.selectbox("Pick your preferred model.", ["random forest", "svm", "xgboost", "decision tree", "naive bayes multinomial", "naive bayes complement", "knn"])

model = Models(choice)
model.final_result()

predict_button = st.button("Predict Poem Category")

if predict_button:
    result = model.clf.predict([text])
    st.text(result[0])
    # st.pyplot(ConfusionMatrixDisplay(confusion_matrix()).plot())
    # st.pyplot(confusion_matrix())
    # st.write(confusion_matrix)
    # confusion_matrix
    st.image("temp.png")
    st.code(model.metrics())
    # st.text(model.metrics())
    # st.metric(model.metrics())
    