from flask import Flask, render_template
import pickle
import pandas as pd

filename = 'recommendation_system.pkl'
rec_system  = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/submit")
def submit():
    ratings = rec_system.loc['02deuce'].sort_values(ascending=False)[0:20]
    return pd.concat([ratings], axis=1).index.values.tolist()

if __name__ == '__main__':
    app.run()
