import streamlit as st
import pickle

import numpy as np
import pandas as pd

tfidf = pickle.load(open('objects/vectorizer.pkl', 'rb'))
model = pickle.load(open('objects/model.pkl', 'rb'))

st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        font-family: 'Times New Roman', Times, serif;
    }
    .main {
        padding: 20px;
    }
    .title {
        color: #031b85;
        text-align: center;
        margin-bottom: 30px;
    }
    .input-section {
        margin: 10px;
        font-family: 'Times New Roman', Times, serif;
    }
    .result-section {
        border-radius: 10px;
        margin: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .footer {
        text-align: center;
        margin-top: 100px;
        font-size: 16px;
        color: black;
    }
    .custom-button:hover {
        background-color: #02135f;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.markdown("<h1 class='title'>Spam Sentinel</h1>", unsafe_allow_html=True)
st.markdown("<div class='input-section'><p>Enter the message:</p></div>", unsafe_allow_html=True)

# User input
user_input = st.text_area("Message")

import re, string

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def rem_stopwords(text):
    words = nltk.word_tokenize(text)
    temp = [word for word in words if word not in stop_words]
    return " ".join([ps.stem(word) for word in temp])

def text_cleaning(text):
    pattern = re.compile('<.*?>')
    text = pattern.sub(r'', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = rem_stopwords(text)
    return text

# Custom button using HTML and CSS
# st.markdown(
#     """
#     <div style='text-align: center;'>
#         <button class='custom-button' onclick="document.getElementById('predict-button').click();">Predict</button>
#     </div>
#     <button id='predict-button' style='display: none;'>Predict</button>
#     """,
#     unsafe_allow_html=True
# )

if st.button('Predict', key='predict-button'):
    if len(user_input.split()) < 5:
        st.warning("Please enter at least 5 words.")
    else:
        msg= text_cleaning(user_input)
        vector_input = tfidf.transform([msg])
        result= model.predict(vector_input)[0]
        prob= model.predict_proba(vector_input)[0]

        spam_prob = prob[1] * 100
        not_spam_prob = prob[0] * 100

        if spam_prob > 0.25:
            st.header("Spam")
        else:
            st.header("Not Spam")

        # st.write(f"Spam Probability: {spam_prob:.2f}% ")

st.markdown("<div class='footer'>Managed by Kishan Periwal</div>", unsafe_allow_html=True)
