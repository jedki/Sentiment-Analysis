import streamlit as st
import streamlit.components.v1 as com
#import libraries
from transformers import AutoModelForSequenceClassification,AutoTokenizer, AutoConfig
import numpy as np
#convert logits to probabilities
from scipy.special import softmax




#import the model
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

model_path = f"Junr-syl/tweet_sentiments_analysis"
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
#Set the page configs
st.set_page_config(page_title='Sentiments Analysis',page_icon='😎',layout='wide')

#welcome Animation
com.iframe("https://embed.lottiefiles.com/animation/149093")
st.markdown('<h1> Tweet Sentiments </h1>',unsafe_allow_html=True)

#Create a form to take user inputs
with st.form(key='tweet',clear_on_submit=True):
    text=st.text_area('Copy and paste a tweet or type one',placeholder='I find it quite amusing how people ignore the effects of not taking the vaccine')
    submit=st.form_submit_button('submit')

#create columns to show outputs
col1,col2,col3=st.columns(3)
col1.title('Sentiment Emoji')
col2.title('How this user feels about the vaccine')
col3.title('Confidence of this prediction')

if submit:
    print('submitted')
    #pass text to preprocessor
    def preprocess(text):
    #initiate an empty list 
        new_text = []
        #split text by space
        for t in text.split(" "):
            #set username to @user
            t = '@user' if t.startswith('@') and len(t) > 1 else t  
            #set tweet source to http
            t = 'http' if t.startswith('http') else t 
            #store text in the list
            new_text.append(t)
            #change text from list back to string
        return " ".join(new_text) 
    

    #pass text to model

    #change label id 
    config.id2label = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}

    text = preprocess(text)

    # PyTorch-based models
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    #Process scores
    ranking = np.argsort(scores)
    ranking = ranking[::-1]  
    l = config.id2label[ranking[0]]
    s = scores[ranking[0]]

    #output
    if l=='NEGATIVE':
        with col1:
            com.iframe("https://embed.lottiefiles.com/animation/125694")
        col2.write('Negative')
        col3.write(f'{s}%')
    elif l=='POSITIVE':
        with col1:
            com.iframe("https://embed.lottiefiles.com/animation/148485")
        col2.write('Positive')
        col3.write(f'{s}%')
    else:
        with col1:
            com.iframe("https://embed.lottiefiles.com/animation/136052")
        col2.write('Neutral')
        col3.write(f'{s}%')


