# Project Title
Computational Tools for Data Science: Movie Recommendation System

# Overview
This project develops a personalized movie recommendation system. It leverages sentiment analysis, clustering, and similarity detection techniques to suggest movies that align with user preferences, while also identifying which streaming platforms they are available on.

# Data sources:
 - TMDB_all_movies.csv: https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates (downloaded 29/10/2024)
 - amazon.csv: https://www.kaggle.com/datasets/octopusteam/full-amazon-prime-dataset (downloaded 06/11/2024)
 - apple.csv: https://www.kaggle.com/datasets/octopusteam/full-apple-tv-dataset (downloaded 06/11/2024) 
 - hbo.csv: https://www.kaggle.com/datasets/octopusteam/full-hbo-max-dataset (downloaded 06/11/2024)
 - hulu.csv: https://www.kaggle.com/datasets/octopusteam/full-hulu-dataset (downloaded 06/11/2024) 
 - netflix.csv: https://www.kaggle.com/datasets/octopusteam/full-netflix-dataset (downloaded 06/11/2024) 

# Usage
The notebook ctds_g6_project.ipynb contains all the code used to build the recommendation system and specifically it comprises:

- retriving data
- data processing
- methodologies which comprends the implementation of clustering, similar items minhashing and sentiment

Note that to run ctds_g6_project.ipynb the TMDB_all_movies.csv dataset is required which, due to it's capacity can't be uploaded to the repository and has to be installed locally.

Once ctds_g6_project.ipynb is run, input a reference movie title to receive 10 similar movie suggestions, categorized by clusters, with streaming platform availability.

# Authors
Raquel Chaves Martinez
Paula Gambus i Moreno
Angel Paisan Garcia
Alba Pi Mas 

 

