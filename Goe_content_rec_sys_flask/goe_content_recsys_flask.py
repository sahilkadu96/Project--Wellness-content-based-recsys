from flask import Flask, render_template, redirect, session
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, SelectField, IntegerField, StringField
from wtforms.validators import DataRequired
import numpy as np
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import jaccard
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(r'C:\Users\Sahil\Data science Machine Learning\Flask_bootcamp\flask_tut\Goe_content_rec_sys_flask\title_features.csv')
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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'MY_SECRET_KEY'

titles = df['Title'].unique()
metrics = ['cosine', 'euclidean', 'pearson', 'manhattan']

class ContentRec(FlaskForm):
    title = SelectField('Select title', choices = titles)
    metric = SelectField('Select metric', choices=metrics)
    number = IntegerField('Enter the number of recommendations', validators=[DataRequired()])
    submit = SubmitField('Recommend')


@app.route('/', methods = ['GET', 'POST'])
def index():
    form = ContentRec()
    if form.validate_on_submit():
        session['title'] = form.title.data
        session['metric'] = form.metric.data
        session['number'] = form.number.data
        return redirect('result')
    return render_template('home.html', form = form)


@app.route('/result', methods = ['GET', 'POST'])
def result():
    res = recommend_similar(title = session['title'], num_to_be_recommended = session['number'], metric = session['metric'])
    return render_template('result.html', tables = [res.to_html()], titles = [''])



if __name__ == '__main__':
    app.run()


