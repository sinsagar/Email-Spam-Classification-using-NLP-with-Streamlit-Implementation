import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Initialize a list to store cleaned and stemmed words
    y = []

    # Remove non-alphanumeric characters and stopwords, and perform stemming
    for i in text:
        if i.isalnum() and i not in stopwords.words('english'):
            y.append(ps.stem(i))

    # Join the cleaned and stemmed words into a string
    return " ".join(y)

vec = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email Spam Classifier")
input_email = st.text_input("Enter the Email")
if st.button('Predict'):
# 1.preprocess

   transform_email = transform_text(input_email)
# 2. Vectorise
   vector_input = vec.transform([transform_email])
# 3.Predict

   result = model.predict(vector_input)[0]
# 4.Display
   if result == 1:
      st.header("Spam")
   else :
        st.header("Not Spam")

