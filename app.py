import numpy as np
from flask import Flask, request, jsonify, render_template
import os
from load import load_model, load_tokenizer, predict_spam

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    msg1=request.form["experience"]

    # pass this through tokenizer first 

    model = load_model()
    tokenizer=load_tokenizer()
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    # prediction = model.predict(msg1)
    pred = predict_spam([msg1], tokenizer, model)

    #return render_template('index.html', prediction_text='Your message  is  '+pred)
    return render_template('index.html', pred_value=pred)


if __name__ == "__main__":
    app.run(debug=True)