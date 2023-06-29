import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.metrics import jaccard_score

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from scipy.spatial.distance import jaccard

import warnings
warnings.filterwarnings('ignore')

st.title('GOE Wellness')
st.header('Content based recommendation')

df = pd.read_csv(r'C:\Users\Sahil\.spyder-py3\title_features.csv')
tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1,3))
tf_matrix = tf.fit_transform(df['feature'])

def get_cos_sim_matrix(tf_matrix):
    return cosine_similarity(tf_matrix, tf_matrix)

def get_euc_dist_matrix(tf_matrix):
    return euclidean_distances(tf_matrix, tf_matrix)

def get_pearson_coef_matrix(tf_matrix):
    tf_matrix_array  = tf_matrix.toarray()
    return np.corrcoef(tf_matrix_array)

def get_manh_dist_matrix(tf_matrix):
    return manhattan_distances(tf_matrix, tf_matrix)


def recommend_similar(title, num_to_be_recommended, metric):
    index = df[df['Title'] == title].index[0]
    
    if metric == 'cosine':
        sim_matrix = get_cos_sim_matrix(tf_matrix)
    elif metric == 'euclidean':
        sim_matrix = get_euc_dist_matrix(tf_matrix)
    elif metric == 'pearson' :
        sim_matrix = get_pearson_coef_matrix(tf_matrix)
    elif metric == 'manhattan':
        sim_matrix = get_manh_dist_matrix(tf_matrix)
    else:
        return "Invalid metric"
        
        
    sims = list(enumerate(sim_matrix[index]))
    
    if metric == 'euclidean' or metric == 'manhattan':
        sims = sorted(sims, key = lambda x:x[1], reverse = False)[0: num_to_be_recommended]
    else:
        sims = sorted(sims, key = lambda x:x[1], reverse = True)[0: num_to_be_recommended]
        
    rec_titles = []
    scores = []
    pillars = []
    instructors = []
    difficulty = []
    duration = []
    
    
    for i in range(0, len(sims)):
        ind = sims[i][0]
        rec_title = df['Title'][ind]
        rec_titles.append(rec_title)
        score = sims[i][1]
        scores.append(score)
        pillars.append(df['Pillar'][ind])
        instructors.append(df['Instructor'][ind])
        difficulty.append(df['Difficulty'][ind])
        duration.append(df['Duration'][ind])
     
        
    res = pd.DataFrame({'Recommended_titles': rec_titles, f'{metric}': scores, 'Pillar': pillars, 'Instructor': instructors,
                       'Difficulty': difficulty, 'Duration': duration})
    return res


titles = df['Title'].unique()
title = st.selectbox('Enter the title', titles)

metrics = ['cosine', 'euclidean', 'pearson', 'manhattan']
metric = st.selectbox('Enter metric', metrics)

number = st.select_slider('Enter the number of recommendations', range(1, 11))

if st.button('Recoomend'):
    res = recommend_similar(title = title, num_to_be_recommended = number, metric = metric)
    st.write(res)