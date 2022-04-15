from flask import Flask,render_template,url_for,request
import pickle
import pandas as pd

# Load recommendation system
rec_system  = pickle.load(open('pickle/recommendation_system.pkl', 'rb'))

# Load vectorizer
tfidf_vectorizer = pickle.load(open('pickle/tfidf_vectorizer.pkl', 'rb'))

# Load sentiment analysis model
sentiment_model = pickle.load(open('pickle/sentiment_analysis.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def submit():
    # ratings = rec_system.loc['02deuce'].sort_values(ascending=False)[0:20]
    # return "#".join(pd.concat([ratings], axis=1).index.values.tolist())
    return render_template('results.html', name='Ankur')

if __name__ == '__main__':
    app.run()
