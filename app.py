from unicodedata import name
from flask import Flask, render_template, url_for, request
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re

# Load recommendation system
rec_system = pickle.load(open('pickle/recommendation_system.pkl', 'rb'))

# Load vectorizer
tfidf_vectorizer = pickle.load(open('pickle/tfidf_vectorizer.pkl', 'rb'))

# Load sentiment analysis model
sentiment_model = pickle.load(open('pickle/sentiment_analysis.pkl', 'rb'))

# Load product name and review text mapping
product_review_map = pickle.load(open('pickle/product_review_map.pkl', 'rb'))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def submit():
    user_name = request.form['username']
    recommended_products = get_recommended_products(user_name)
    return render_template('results.html', products=recommended_products)


def get_recommended_products(user_name):

    # Top 20 products from recommendation system
    ratings = rec_system.loc[user_name].sort_values(ascending=False)[0:20]
    products = pd.concat([ratings], axis=1).index.values.tolist()

    product_prediction_dict = dict()

    for product_name in products:

        # Get the product reviews
        product_reviews = get_reviews_for_product(
            product_name, product_review_map)

        # Get reviews after preprocessing
        processed_reviews_text = list(map(pre_process_text, product_reviews))

        # Transform reviews data using vectorizer
        transformed_reviews = tfidf_vectorizer.transform(
            processed_reviews_text)

        # Predict sentiment
        predictions = list(sentiment_model.predict(transformed_reviews))

        pos_predictions_perc = (sum(predictions)/len(predictions)) * 100

        product_prediction_dict[product_name] = pos_predictions_perc

    top_five_recommendations = {k: product_prediction_dict[k] for k in sorted(
        product_prediction_dict, key=product_prediction_dict.get, reverse=True)[:5]}

    return top_five_recommendations


def get_reviews_for_product(product_name, product_review_map):
    return list(product_review_map[product_review_map['name'] == product_name]['reviews_text'])


def pre_process_text(review_text):
    stop = stopwords.words('english')
    review_text = ' '.join([word for word in review_text.split(
    ) if word not in (stop)])  # stop words removal
    review_text = review_text.lower()  # changing text to lower case
    review_text = review_text.translate(str.maketrans(
        '', '', string.punctuation))  # removing punctuation
    # removing digits and other characters
    review_text = re.sub("(\\W|\\d)", " ", review_text)
    review_text = lemmatize_text(review_text)  # Lemmatization
    return review_text


def lemmatize_text(review_text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(review_text)
    lemmatized_output = ' '.join(
        [lemmatizer.lemmatize(token) for token in tokens])
    return lemmatized_output


if __name__ == '__main__':
    app.run()
