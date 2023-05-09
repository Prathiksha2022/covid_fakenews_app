import pickle
from flask import Flask, request, jsonify
import re
import nltk
nltk.download('stopwords')
from nltk import PorterStemmer
from nltk.corpus import stopwords
ps = PorterStemmer()

# Load the logistic regression model from the pkl file
with open('model2.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the CountVectorizer from the pkl file
with open('tfidfvect2.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define a function to preprocess the input news text
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', str(text))
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = vectorizer.transform([review]).toarray()
    prediction = model.predict(review_vect)
    return prediction

# Initialize the Flask app
app = Flask(__name__)

# Define an API endpoint for classifying news as real or fake
@app.route('/classify_news', methods=['POST'])
def classify_news():
    # Get the news text from the POST request
    data = request.get_json()
    news = data['news']
    pred = preprocess_text(news)
    if pred[0] == 1:
        return jsonify({'prediction': '1'})
    else:
        return jsonify({'prediction': '0'})

# Run the Flask app
if __name__ == '__main__':
    app.run()
