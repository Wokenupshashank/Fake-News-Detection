import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re #regular extraction is function which determine wheter a given text fits in the expression 
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
vectorization=TfidfVectorizer()

vector_form=pickle.load(open("vector.pkl","rb"))
load_model=pickle.load(open("model.pkl","rb"))

def wordopt(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text) #sub means sub strain. it will following thins in the data sets  
    text=re.sub("\\W"," ",text)
    text=re.sub('https?://\S+|www\.|S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]'% re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    return text
 #it will remove above like special character from dataset

def fake_news(news):
    news=wordopt(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction=load_model.predict(vector_form1)
    return prediction

if __name__=="__main__":
    st.title('Fake News Classification app ')
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "Some news", height=200)
    predict_btt = st.button("predict")
    if predict_btt:
        prediction_class=fake_news (sentence)
        print (prediction_class)
        if prediction_class == [0]:
            st. success ('Unreliable or fake new.....')
        if prediction_class == [1]:
            st.warning('Reliable or Real news...')

