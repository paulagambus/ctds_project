## This document contains the functions used in in the project


import string
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import numpy as np

from datasketch import MinHash, MinHashLSH

#Functions to preprocess the data
def remove_tv_show(df):
    return df[df['type'] == 'movie']


def platform_column(main, platform, platform_name):
    #add platform column to main dataset
    for index, row in main.iterrows():
        if row['title'] in platform['title'].values: 
            main.loc[index, platform_name] = 1
        else:
            main.loc[index, platform_name] = 0
    return main

# Functions to analyze the data
def count_movies(df, platform_name):
    return df[platform_name].sum()

def more_than_one_platform(df):
    #return the movies that are available on more than one platform
    return df[(df['Netflix'] + df['Amazon'] + df['Hulu'] + df['Apple'] + df['HBO']) > 1]


#Functions to calculate the sentiment
def sentiment(tokens):
    #load the LabMT wordlist from Data_Set_S1.txt
    with open("./data/Data_Set_S1.txt") as f:
        lines = f.readlines()
    #remove the first 4 lines 
    lines = lines[4:]

    #create a dictionary of words and their happiness values
    word_dict = {}
    for line in lines:
        line = line.split('\t')
        word = line[0]
        happiness = float(line[2])
        word_dict[word] = happiness

    #calculate the sentiment of the tokens
    sentiment = 0
    for token in tokens:
        if token in word_dict:
            sentiment += word_dict[token]
    return round(sentiment, 2)

def preprocess_and_analyze_sentiment(text, lemmatizer = WordNetLemmatizer(), stop_words = set(stopwords.words('english')), punctuation_table = str.maketrans('', '', string.punctuation)):
    # Check for non-null and non-'NaN' string types
    if isinstance(text, str) and text != 'NaN':
        # Lowercase, remove punctuation, tokenize, remove stopwords, and lemmatize
        text = text.lower().translate(punctuation_table)
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
        
        # Calculate sentiment score
        return sentiment(tokens)
    else:
        # Return NaN if the text is not valid
        return np.nan
    
def categorize_sentiment(value, low_threshold, high_threshold):
    if value < low_threshold:
        return 'low'
    elif value > high_threshold:
        return 'high'
    else:
        return 'medium'
    
# Functions to calculate similarity
def create_combined_minhash(movie, one_hot_features, non_one_hot_features, num_perm=128):
    m = MinHash(num_perm=num_perm)
    
    # Add one-hot encoded features
    for feature in one_hot_features:
        if movie[feature] == 1:
            m.update(feature.encode('utf8'))
    
    # Add non-one-hot encoded features
    for feature in non_one_hot_features:
        for token in movie[feature]:
            m.update(token.encode('utf8'))
    
    return m