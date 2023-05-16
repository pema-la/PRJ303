from flask import Flask, render_template,jsonify
import joblib
from flask import Flask, render_template, redirect, url_for, request, send_from_directory
from flask import Flask, render_template, request, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename
from keras_preprocessing import image
from keras.models import load_model
import sys
import os
import glob
import numpy as np



# Load the trained disease prediction model
random = RandomForestClassifier()
# model.load('rf_model.pkl')
import pickle
# Load the pickle model from a file
with open('random_m20.pkl', 'rb') as f:
    model5 = pickle.load(f)

import csv
# Load the diagnoses and treatments from a CSV file
diagnoses = {}
with open('finalOutput.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        diagnoses[row['diagnosis']] = [row['description'], row['treatment']]


filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
model1 = pickle.load(open('heartdisease_model.sav', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

app = Flask(__name__)  #instance of flask
#app.debug = True
server = app.server

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    intent = req['queryResult']['intent']['displayName']
    symptom1 = req['queryResult']['parameters']['symptom1']
    symptom2 = req['queryResult']['parameters']['symptom2']
    symptom3 = req['queryResult']['parameters']['symptom3']
    symptom4 = req['queryResult']['parameters']['symptom4']
    symptom5 = req['queryResult']['parameters']['symptom5']
    symptom6 = req['queryResult']['parameters']['symptom6']
    symptom7 = req['queryResult']['parameters']['symptom7']
    symptom8 = req['queryResult']['parameters']['symptom8']
    symptom9 = req['queryResult']['parameters']['symptom9']
    symptom10 = req['queryResult']['parameters']['symptom10']
    symptom11 = req['queryResult']['parameters']['symptom11']
    symptom12 = req['queryResult']['parameters']['symptom12']
    symptom13 = req['queryResult']['parameters']['symptom13']
    symptom14 = req['queryResult']['parameters']['symptom14']
    symptom15 = req['queryResult']['parameters']['symptom15']
    symptom16 = req['queryResult']['parameters']['symptom16']
    symptom17 = req['queryResult']['parameters']['symptom17']
    symptom18 = req['queryResult']['parameters']['symptom18']
    symptom19 = req['queryResult']['parameters']['symptom19']
    symptom20 = req['queryResult']['parameters']['symptom20']
    
    # Call your disease prediction model with the input symptoms
    diagnosis,description, treatment = predict_disease(symptom1, symptom2, symptom3, symptom4, symptom5,symptom6, symptom7, symptom8, symptom9, symptom10,symptom11, symptom12, symptom13, symptom14, symptom15,symptom16, symptom17, symptom18, symptom19, symptom20)

    # Generate a response to send back to the user
    response = {
        "fulfillmentMessages": [
            {
                "text": {
                    "text": [
                        "Based on your symptoms, it looks like you have " + diagnosis +". \n\nKnow more about " + diagnosis + ": \n " + description + ". \n\nI recommend the following treatment: \n" + treatment + "."
                    ]
                }
            }
        ]
    }

    return jsonify(response)

def predict_disease(symptom1, symptom2, symptom3, symptom4, symptom5, symptom6, symptom7, symptom8, symptom9, symptom10, symptom11, symptom12, symptom13, symptom14, symptom15, symptom16, symptom17, symptom18, symptom19, symptom20):
    # Convert the input symptoms to a feature vector
    features = [symptom1, symptom2, symptom3, symptom4, symptom5, symptom6, symptom7, symptom8, symptom9, symptom10, symptom11, symptom12, symptom13, symptom14, symptom15, symptom16, symptom17, symptom18, symptom19, symptom20]
  
    diagnosis = model.predict([features])[0]

    if diagnosis in diagnoses:
        description = diagnoses[diagnosis][0]
        treatment = diagnoses[diagnosis][1]
       
    else:
        treatment = 'No treatment recommended'
        description = 'No description available'

    return diagnosis,description, treatment



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['brainimg'],filename)

@app.route("/")  
def base():
    return render_template('home.html')

@app.route("/service")
def service():
    return render_template('service.html')

@app.route("/about")
def about():
    return render_template('about.html') 
  
@app.route("/contact")
def contact():
    return render_template('contact.html')

@app.route("/form")
def form():
    return render_template('form.html')

@app.route("/BreastC")
# @login_required
def cancer():
    return render_template("BreastC.html")

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
                     'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    if output == 4:
        res_val = "a high risk of Breast Cancer"
        coment="Keep a healthy weight. Be physically active. Choose not to drink alcohol, or drink alcohol in moderation. If you are taking, or have been told to take, hormone replacement therapy or oral contraceptives (birth control pills), ask your doctor about the risks and find out if it is right for you."
    else:
        res_val = "a low risk of Breast Cancer"
        coment="You should limit alcohol, Maintain a healthy weight, Be physically active,Breast-feed dan Limit postmenopausal hormone therapy. Maintaining a healthy weight also is a key factor in breast cancer prevention."

    return render_template('breastC_result.html', prediction_text='Patient has {}'.format(res_val), coment=coment)


@app.route("/diabetes")
# @login_required
def diabetes():
    return render_template("diabetes.html")

##################################################################################

df1 = pd.read_csv('diabetes.csv')

# Renaming DiabetesPedigreeFunction as DPF
df1 = df1.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df1.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure',
                                                                                    'SkinThickness', 'Insulin',
                                                                                    'BMI']].replace(0, np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# Model Building

X = df1.drop(columns='Outcome')
y = df1['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model

classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

#####################################################################


@app.route('/predictt', methods=['POST'])
def predictt():
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)

        return render_template('dia_result.html', prediction=my_prediction)


############################################################################################################

@app.route("/heart")
# @login_required
def heart():
    return render_template("heart.html")

@app.route('/predictheart', methods=['POST'])
def predictheart():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

   
    features_name = ["age", "sex", "cp", "thalach","exang", "oldpeak", "slope", "ca" ,"thal"]

    df = pd.DataFrame(features_value, columns=features_name)
    output = model1.predict(df)

    if output == 1:
        res_val = "a high risk of Heart Disease"
        recommend="Regular physical activity can lower your risk for heart disease. Drinking too much alcohol can raise blood pressure levels and the risk for heart disease. It also increases levels of triglycerides, a fatty substance in the blood which can increase the risk for heart disease."
    else:
        res_val = "a low risk of Heart Disease"
        recommend ="Be sure to eat plenty of fresh fruits and vegetables and fewer processed foods. Eating lots of foods high in saturated fat and trans fat may contribute to heart disease. Eating foods high in fiber and low in saturated fats, trans fat, and cholesterol can help prevent high cholesterol."

    return render_template('heart_result.html', Heart_prediction_text='Patient has {}'.format(res_val),recommend=recommend)




model4 = load_model("BT.h5")


def model_predict(img_path, model4):
    img = image.load_img(img_path, target_size=(200,200)) #target_size must agree with what the trained model expects!!

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')/255
   
    preds = model4.predict(img)

   
   
    pred = np.argmax(preds,axis = 1)
    return pred


@app.route('/BrainTumor', methods=['GET'])
def BrainTumor():
    return render_template("BrainTumor.html")


@app.route('/Bpredict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        pred = model_predict(file_path, model4)
        os.remove(file_path)#removes file from the server after prediction has been returned

        # Arrange the correct return according to the model. 
		# In this model 1 is Pneumonia and 0 is Normal.
        str0 = 'Base on your MRI image it looks like you are suffering from Glioma Brain Tumor'
        str1 = 'Base on your MRI image it looks like you are suffering from Meningioma Brain Tumor'
        str3 = 'Base on your MRI image it looks like you are suffering from pituitary Brain Tumor'
        str2 = 'Base on your MRI image it looks like you are not suffering from Brain Tumour'
        if pred[0] == 0:
            return str0
        elif pred[0] == 1:
            return str1
        elif pred[0]==3:
            return str3
        else:
            return str2
    return None


@app.route("/learn")
# @login_required
def learn():
    return render_template("LearnDi.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
