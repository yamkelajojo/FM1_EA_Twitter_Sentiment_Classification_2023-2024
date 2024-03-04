"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
import joblib,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
import string
from nltk import SnowballStemmer, PorterStemmer, LancasterStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import warnings
import pickle
from sklearn.feature_extraction.text import CountVectorizer
warnings.filterwarnings("ignore")




# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = joblib.load("betterVect_sentiment.pkl","rb")
#tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("train.csv")
#tweet_text = st.text_area("Enter Text","Type Here")
# The main function where we will build the actual app
def main(raw=raw):
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown(" Data description as per source")
		st.text("Data description as per source The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo.This dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were annotated. Each tweet is labelled independently by 3 reviewers. This dataset only contains tweets that all 3 reviewers agreed on (the rest were discarded).Each tweet is labelled as one of the following classes:- 2(News): the tweet links to factual news about climate change- 1(Pro): the tweet supports the belief of man-made climate change- 0(Neutral): the tweet neither supports nor refutes the belief of man-made climate change- -1(Anti): the tweet does not believe in man-made climate change")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
	tweet_text = st.text_area("Enter Text","Type Here")
	if selection == "Predictions":
		st.info("Prediction with ML Models")
		
	if st.button("Classify"):
		text = list([tweet_text])
		raw =pd.DataFrame(text, columns=["message"])
		raw['message'].replace('\d+', '', regex = True, inplace= True)

		def remove_RT(column_name):
    			return re.sub(r'^rt[^\s]+', '',column_name)
		raw['message']= raw['message'].apply(remove_RT)
		raw['message'] = raw['message'].str.replace('rt', '')
		def remove_handels(post):
    			return re.sub('@[^\s]+',' ',post)
		raw['message']= raw['message'].apply(remove_handels)
				#removing the url
		pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
		raw['message'] = raw['message'].replace(to_replace = pattern_url,value = " ", regex = True)
		def remove_hashtages(post):
    			return re.sub('#[^\s]+',' ',post)
		raw['message']= raw['message'].apply(remove_hashtages)
		def remove_punctuation(post):
    			return ''.join([l for l in post if l not in string.punctuation])
		raw["message"] = raw["message"].apply(remove_punctuation)
		raw['message'] = raw['message'].str.split()
		stemmer = SnowballStemmer("english")
		raw['message'] = raw['message'].apply(lambda x: [stemmer.stem(y) for y in x])
			
		def remove_stop_words(tokens):    
    			return [t for t in tokens if t not in stopwords.words('english')]
		raw['message'] = raw['message'].apply(remove_stop_words)
		nltk.download('wordnet')
		lemmatizer = WordNetLemmatizer()
		def mbti_lemma(words, lemmatizer):
    			return [lemmatizer.lemmatize(word) for word in words] 
		raw['message'] = raw['message'].apply(mbti_lemma, args=(lemmatizer, ))
		raw["message"] = raw["message"].apply(' '.join)



			
		#tweet_text = raw["message"][0]

    			
	#if st.button("submit"):
		#tweet_text = st.text_area("happy")
		tweet_text = tweet_text.lower()	
		vect_text = news_vectorizer.transform(np.array([tweet_text]))
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
		predictor = joblib.load(open(os.path.join("Climant_change_sentiment.pkl"),"rb"))
		prediction = predictor.predict(vect_text)[0]
		st.success(prediction)
			

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
		st.success("Text Categorized as: {}".format(prediction))
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
