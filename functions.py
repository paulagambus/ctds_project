## This document contains the functions used in in the project


import string
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import numpy as np
import pandas as pd

from datasketch import MinHash, MinHashLSH

from tqdm import tqdm

#Functions to preprocess the data
def remove_tv_show(df):
    """
    This function removes the TV shows from the dataset
    df: dataset with the movies and TV shows
    """
    return df[df['type'] == 'movie']


def platform_column(main, platform, platform_name):
    """
    This function adds a column to the main dataset that indicates whether a movie is available on a specific platform.
    main: dataset with the movies
    platform: dataset with the movies available on the platform
    platform_name: name of the platform
    """
    for index, row in main.iterrows():
        if row['title'] in platform['title'].values: 
            main.loc[index, platform_name] = 1
        else:
            main.loc[index, platform_name] = 0
    return main

# Functions to analyze the data
def count_movies(df, platform_name):
    """
    This function counts the number of movies available on a specific platform.
    df: dataset with the movies
    platform_name: name of the platform
    """
    return df[platform_name].sum()


#Functions to calculate the sentiment
def sentiment(tokens):
    """
    This function calculates the sentiment of a list of tokens.
    tokens: list of tokens
    """
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
    """
    This function preprocesses the text and calculates the sentiment score.
    text: text to be analyzed
    lemmatizer: lemmatizer object
    stop_words: set of stopwords
    punctuation_table: punctuation tables
    """
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
    """
    This function categorizes the sentiment score into 'low', 'medium', or 'high'.
    value: sentiment score
    low_threshold: low threshold
    high_threshold: high threshold
    """
    if value < low_threshold:
        return 'low'
    elif value > high_threshold:
        return 'high'
    else:
        return 'medium'
    
# Functions to calculate similarity
def create_combined_minhash(movie, one_hot_features, non_one_hot_features, num_perm=128):
    """
    This function creates a MinHash object for a movie using the one-hot encoded and non-one-hot encoded features.
    movie: movie data
    one_hot_features: one-hot encoded features
    non_one_hot_features: non-one-hot encoded features
    """
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

def estimate_jaccard_similarity(data, minhashes, movie_title):
    """
    This function estimates the Jaccard similarity between a movie and all other movies in the dataset.
    data: dataset with the movies
    minhashes: MinHash objects for all movies

    """
    # Find all movies with the specified title
    matching_movies = data[data['title'] == movie_title]

    if len(matching_movies) > 1:
        print(f"There are multiple movies with the title '{movie_title}'. Please choose one by its id:")
        for idx, row in matching_movies.iterrows():
            print(f"ID: {row['id']}, Crew: {row['crew']}")
        movie_id = int(input("Enter the movie id: "))
        movie_index = data[data['id'] == movie_id].index[0]
        print(f"Selected movie with id '{movie_id}' in position '{movie_index}'")
    elif len(matching_movies) == 1:
        movie_index = data[data['title'] == movie_title].index[0]
        print('movie_index:', movie_index)
        movie_id = data[data['title'] == movie_title]['id'].values[0]
        print(f"movie_id: {movie_id}")
    else:
        print(f"No movies found with the title '{movie_title}'")
        return []

    # Estimate Jaccard similarity with the specified movie
    print(f"Estimated Jaccard Similarity with '{movie_title}':")
    similarity_scores = []

    for i in tqdm(range(len(minhashes)), desc="Calculating Similarities"):
        if i != movie_index:
            similarity = data.iloc[movie_index]['minhash'].jaccard(data.iloc[i]['minhash'])
            similarity_scores.append((data.iloc[i]['title'], similarity, minhashes[i], data.iloc[i]['id']))

    # Sort the similarity scores in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    return similarity_scores, movie_id

def create_top_n_dataframe(similarity_scores, data, movie_id, n=10):
    """
    This function creates a dataframe with the top N most similar movies.
    similarity_scores: list of similarity scores
    data: dataset with the movies
    movie_id: id of the movie
    n: number of movies to return
    """
    columns = ['title', 'crew', 'production_companies', 'production_countries', 'sentiment_category', 'Netflix', 'Amazon', 'Hulu', 'Apple', 'HBO']
    top_n_df = pd.DataFrame(columns=columns)
    top_n_list = []
    
    for movie, score, minhash, id in similarity_scores[:n]:
        movie_data = data[data['id'] == id]
        top_n_list.append(movie_data)

    top_n_list.append(data[data['id'] == movie_id])
    top_n_df = pd.concat(top_n_list)
    
    return top_n_df

def calculate_jaccard_similarity_matrix(top_10_df):
    """
    This function calculates the Jaccard similarity matrix between the top 10 movies.
    top_10_df: dataframe with the top 10 movies
    """

    jaccard_similarity_matrix = np.zeros((len(top_10_df), len(top_10_df)))
    for i in range(len(top_10_df)):
        for j in range(len(top_10_df)):
            jaccard_similarity = top_10_df.iloc[i]['minhash'].jaccard(top_10_df.iloc[j]['minhash'])
            jaccard_similarity_matrix[i][j] = jaccard_similarity
    return jaccard_similarity_matrix