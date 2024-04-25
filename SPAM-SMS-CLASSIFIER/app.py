# Importing essential libraries
from flask import Flask, render_template, request
import pickle
#from data import data  # You might need to adjust this import based on your project structure

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'spam-sms-mnb-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        vect = cv.transform([message]).toarray()  # Transform the input message
        my_prediction = classifier.predict(vect)
    return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
