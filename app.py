import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
## Load the model
# regmodel=pickle.load(open('regmodel.pkl','rb'))
# scalar=pickle.load(open('scaling.pkl','rb'))

with open('pipeline_model.pkl', 'rb') as file:
    pipeline = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the JSON data from the POST request
    input_data = request.get_json()
    print("API hIT SUCCESS.")
    print(input_data)
    #Convert string values to appropriate data types
    input_data = {k: float(v) if not isinstance(v, float) else v for k, v in input_data.items()}

    # # Create a DataFrame from the input_data dictionary
    df = pd.DataFrame.from_dict([input_data])

    # # Perform inference using the loaded pipeline
    predictions = pipeline.predict(df.values)

    # # Return the predictions as a JSON response
    # response = {'predictions': predictions.tolist()}  # Convert predictions to a list
    # print(response, type(response))
    return jsonify({"response":int(predictions[0])})


# @app.route('/predict', methods=['POST'])
# def predict():
#     # Retrieve the form data
#     male = float(request.form['male'])
#     print("male : ",male)
#     age = float(request.form['age'])
#     BPMeds = float(request.form['BPMeds'])
#     prevalentStroke = float(request.form['prevalentStroke'])
#     prevalentHyp = float(request.form['prevalentHyp'])
#     diabetes = float(request.form['diabetes'])
#     totChol = float(request.form['totChol'])
#     sysBP = float(request.form['sysBP'])
#     diaBP = float(request.form['diaBP'])
#     BMI_ = float(request.form['BMI'])
#     heartRate = float(request.form['heartRate'])
#     glucose = float(request.form['glucose'])
#     input_data = np.array([0,male, age, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI_, heartRate, glucose]).reshape(1, -1)

#         # Perform inference using the pipeline
#     predictions = pipeline.predict(input_data)
#     # predictions = pipeline.predict(np.array([male,age,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI_,heartRate,glucose]).reshape(-1, 1))
#     # Perform inference using the form data
#     # Include your ML prediction logic here
#     # prediction = 0  # Replace with your prediction

#     # Prepare the response
#     print(predictions)
#     response = {'prediction': str(predictions[0])}

#     # Return the response as JSON
#     return render_template("home.html",prediction_text="The House price prediction is {}".format(str(predictions[0])))




if __name__=="__main__":
    app.run(debug=True)
   
     
