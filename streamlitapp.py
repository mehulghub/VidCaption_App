#Import all the dependencies
import streamlit as st
import os
import imageio
import ffmpeg
import numpy as np

import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

#set the layout to the streamlit app as wide
st.set_page_config(layout='wide')

#setup the sidebar
with st.sidebar:
    st.image('https://cdn.dribbble.com/users/6109291/screenshots/18043502/media/db5951c7d84010f1ef02e9c34d016671.png?compress=1&resize=400x300')
    st.title('Welcome to the VidCap AI')
    st.info('This is a simple application that uses a trained model to predict the caption in a video.')



#Generating a list of options or videos
options= os.listdir(os.path.join('..','data','s1'))
print(options)

#Selecting the option from the list
selected_video = st.selectbox('Select a video', options)

col1, col2= st.columns(2)

if options:
    #Rendering the video
    with col1:
        file_path= os.path.join('..','data','s1',selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
     
        #Rendering inside of the app
        video=open('test_video.mp4','rb').read() 
        st.video(video)
    
    with col2:
        st.info('All what the ML model sees when making Prediction')
        video, annotations= load_data(tf.convert_to_tensor(file_path))
        #imageio.mimsave('animation.gif', (video[0]* 255).astype(np.uint8), duration=100)
        imageio.mimsave('animation.gif', tf.squeeze(video)*255, duration=100)
        st.image('animation.gif', width=400)

        st.info('This is the output of ML Model')
        model= load_model()
        yhat= model.predict(tf.expand_dims(video, axis=0))
        #st.text(tf.argmax(yhat, axis=1))
        decoder= tf.keras.backend.ctc_decode(yhat, [75], greedy= True)[0][0].numpy()
        st.text(decoder)


        st.info('Decode the raw tokens into words')
        converted_prediction= tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        #Num to Char


        
    
   

    