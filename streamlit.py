import streamlit as st
from textblob import TextBlob
from transformers import pipeline



st.title('Text Sentiment Analysis')
st.text('This app was created by Ashwin Philip George for the module CET023.')
st.text('It simply takes in an input text and determines the sentiment(positive/negative) of the given text')

classifier =  pipeline('sentiment-analysis', "mrm8488/bert-small-finetuned-squadv2")

speech = st.text_input('Enter your text here!')


if st.button('Predict'):
    r = TextBlob(speech).sentiment
    if r.polarity > 0:
        st.write('Textblob predicts a positive sentiment with score: ' + str(r.polarity))
    else:
        st.write('Textblob predicts a negative sentiment with score: ' + str(r.polarity))
    r_transformer = classifier(speech)
    
    if r_transformer[0]['score']:
        st.write('Textblob predicts a positive sentiment with score: ' + str(r_transformer[0]['score']))
    else:
        st.write('Textblob predicts a negative sentiment with score: ' + str(r_transformer[0]['score']))

