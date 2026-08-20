from flask import Flask, request
import pickle

app = Flask(__name__)

@app.route('/prediction', methods=['POST'])
def preds():
    sl = float(request.form['sl'])
    sw = float(request.form['sw'])
    pl = float(request.form['pl'])
    pw = float(request.form['pw'])

    with open('iris_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    prediction = model.predict([[sl, sw, pl, pw]])
    return f"Predicted value is {prediction}"

app.run()

