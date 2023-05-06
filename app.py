from flask import Flask, request, jsonify, render_template
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')
ps = PorterStemmer()
# Load model and vectorizer
model = pickle.load(open('model2.pkl', 'rb'))
tfidfvect = pickle.load(open('tfidfvect2.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello world"


@app.route('/predict', methods=['GET','POST'])
def predict():
    text = request.form.get('text')
    review = re.sub('[^a-zA-Z]', ' ', str(text))
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    if model.predict(review_vect) == 0:
        prediction = 'REAL'
    else:
        prediction = 'FAKE'
    return jsonify({'prediction': prediction})
    # return prediction



if __name__ == "__main__":
    app.run()
