# Movie-recommendations-system-

A simple Movie Recommendation System that suggests movies based on user preferences. This project uses collaborative filtering and content-based filtering techniques to recommend movies. Built with Python, it leverages popular libraries such as pandas, numpy, scikit-learn, and possibly a machine learning framework like TensorFlow or PyTorch if deep learning models are applied.

import numpy as np
import pandas as pd
import ast

credits_df = pd.read_csv("credits.csv")
movies_df = pd.read_csv("movies.csv")

credits_df
movies_df

movies_df.info()

movies_df = movies_df[['movie_id', 'title' ,'overview', 'genres', 'cast', 'crew', 'keywords']] 
movies_df.head()

movies_df.info()

movies_df.isnull().sum()

movies_df.dropna(inplace = True)

movies_df.duplicated().sum()

movies_df.iloc[0].genres

def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name']) 
    return L   

movies_df['genres']= movies_df['genres'].apply(convert)
movies_df['keywords']= movies_df['keywords'].apply(convert)
movies_df.head()

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter +=1
        else:
            break
        return L    

movies_df['cast'] = movies_df['cast'].apply(convert3)

movies_df.head()

def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
    return L

movies_df['crew']=movies_df['crew'].apply(fetch_director)

movies_df

movies_df['overview'][0]

movies_df['overview']=movies_df['overview'].apply(lambda x:x.split())

movies_df


movies_df['genres']=movies_df['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies_df['keywords']=movies_df['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies_df['crew']=movies_df['crew'].apply(lambda x:[i.replace(" ", "") for i in x])

movies_df['tags'] = movies_df['overview']+movies_df['genres']+movies_df['keywords']+movies_df['crew']

movies_df

new_df = movies_df[['movie_id' , 'title', 'tags']]

new_df

new_df['tags'][0]

new_df['tags']= new_df['tags'].apply(lambda X:X.lower())

new_df.head()

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 5000, stop_words= 'english')

cv.fit_transform(new_df['tags']).toarray().shape

vectors = cv.fit_transform(new_df['tags']).toarray()

vectors[0]

import nltk

from  nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)    

new_df['tags']= new_df['tags'].apply(stem)

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity(vectors)

cosine_similarity(vectors).shape

similarity = cosine_similarity(vectors)

similarity[0]

similarity[0].shape

sorted(list(enumerate(similarity[0])), reverse= True, key = lambda x:x[1])[1:6]

def recommend(movie):
    movie_index = new_df[new_df['title']==movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    
  for i in movies_list: 
        print(new_df.iloc[i[0]].title)

recommend('Avatar')


    

    

