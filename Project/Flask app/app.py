from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


@app.route('/') #rendering the html template
def home():
    return render_template('home.html')

@app.route("/predict", methods=["GET", "POST"])
def predict():
    AT = float(request.form['AT'])
    V = float(request.form['V'])
    AP = float(request.form['AP'])
    RH = float(request.form['RH'])
    #converting
    data = [[float(AT), float(V), float(AP), float(RH)]]
    #Loading model
    model = pickle.load(open('CCPP.pkl', 'rb'))
    prediction = model.predict(data)[0]
    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
