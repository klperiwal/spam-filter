import streamlit as st
import pickle
import pandas as pd

# Load the model and vectorizer
tfidf = pickle.load(open('objects/vectorizer.pkl', 'rb'))
model = pickle.load(open('objects/model.pkl', 'rb'))

# Custom CSS for background and text styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main {
        padding: 20px;
    }
    .title {
        font-family: 'Arial Black', sans-serif;
        color: #4B0082;
        text-align: center;
        margin-bottom: 50px;
    }
    .input-section, .result-section {
        margin: 50px 0;
        padding: 20px;
        background-color: #FFFFFF;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 14px;
        color: #4B0082;
    }
    .btn-primary {
        background-color: #4B0082;
        border-color: #4B0082;
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

import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

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

if st.button('Predict'):
    if len(user_input.split()) < 5:
        st.warning("Please enter at least 5 words.")
    else:
        msg = text_cleaning(user_input)
        vector_input = tfidf.transform([msg])
        result = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)[0]

        spam_proba = proba[1] * 100
        not_spam_proba = proba[0] * 100

        if result == 1:
            st.header("Spam")
            st.write(f"There is a {spam_proba:.2f}% chance that this message is spam.")
        else:
            st.header("Not Spam")
            st.write(f"There is a {not_spam_proba:.2f}% chance that this message is not spam.")

# Footer
st.markdown("<div class='footer'>Managed by Kishan Periwal</div>", unsafe_allow_html=True)
