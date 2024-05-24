import streamlit as st
from models import Models

st.title("Classifying Poems")

text = st.text_input("Enter the text to classify.")
choice = st.selectbox("Pick your preferred model.", ["random forest", "svm", "xgboost", "decision tree", "naive bayes multinomial", "naive bayes complement", "knn"])

model = Models(choice)
model.final_result()

predict_button = st.button("Predict Poem Category")

if predict_button:
    result = model.clf.predict([text])
    st.text(result[0])
    st.image("temp.png")
    st.code(model.metrics())
    