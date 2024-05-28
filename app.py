import streamlit as st
from models import Models
import streamlit.components.v1 as components

# col1, col2, col3 = st.columns(3)

st.markdown("<h1 style='text-align: center; color: white;'>Classifying Poems</h1>", unsafe_allow_html=True)

# with col1:
#     st.write("")
# with col3:
#     st.write("")
# with col2:
#     st.title("Classifying Poems")

text = st.text_input("Enter the text to classify.")
choice = st.selectbox("Pick your preferred model.", ["random forest", "svm", "xgboost", "decision tree", "naive bayes multinomial", "naive bayes complement", "knn"])

model = Models(choice)
model.final_result()

predict_button = st.button("Predict Poem Category")

if predict_button:
    if choice == "xgboost":
        classes = model.xg_boost()[2]
        result = classes[model.clf.predict([text])]
    else:
        result = model.clf.predict([text])
    st.text(result[0])
    
    # with col1:
    st.image("temp.png")
    
    # col1, col2, col3 = st.columns(3)
    
    # with col2:
    st.dataframe(model.metrics())
    # components.html('<div style="text-align: center"> st.dataframe(model.metrics()) </div>')
    # <div style="text-align: center"> st.dataframe(model.metrics()) </div>

# Solution provided by dataprofessor (https://discuss.streamlit.io/t/image-in-markdown/13274/10) modified by mze3e to center the image
# img_to_bytes and img_to_html inspired from https://pmbaumgartner.github.io/streamlitopedia/sizing-and-images.html

# import base64
# from pathlib import Path

# def img_to_bytes(img_path):
#     img_bytes = Path(img_path).read_bytes()
#     encoded = base64.b64encode(img_bytes).decode()
#     return encoded
# def img_to_html(img_path):
#     img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
#       img_to_bytes(img_path)
#     )
#     return img_html

# st.markdown(<p style='text-align: center; color: grey;'>"+img_to_html('image.png')+"</p>", unsafe_allow_html=True)
    