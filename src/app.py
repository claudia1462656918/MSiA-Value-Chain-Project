import logging.config
import pickle
import traceback
import xgboost
import pandas as pd
import numpy as np

from datetime import datetime
from flask import Flask,render_template, url_for, flash, redirect,request
import logging.config
# from app import db, app

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from src.database_model import User

app = Flask(__name__)
app.config.from_object('config')

db = SQLAlchemy(app)



logging.config.fileConfig(app.config["LOGGING_CONFIG"])
logger = logging.getLogger("penny-lane")
logger.debug('Test log')


@app.route("/index")
@app.route("/")
def index():
    return render_template('index.html', title='Index')


@app.route("/contact", methods=['POST','GET'])
def contact():
    return render_template('contact.html', title='contact')


@app.route("/prediction")
def prediction():
    return render_template('prediction.html', title='Prediction')

@app.route("/user")
def user():
    return render_template('user.html', title='User')


@app.route("/advice")
def advice():
    return render_template('elements.html', title='Advice')


def ValuePredictor(to_predict_list):

    to_predict = np.array(to_predict_list).reshape(1,15)
    loaded_model = pickle.load(open("models/bank-prediction.pkl","rb"))
    features_columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing','loan', 'contact', 'day', 'month', 'campaign', 'pdays', 'previous','poutcome']
    new_df = pd.DataFrame(to_predict, columns = features_columns)
    result = loaded_model.predict(new_df)
    return result[0]


@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':


        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list)) 
        result = ValuePredictor(to_predict_list)

        if int(result)==1:
            prediction='Buy It! The Time Is NOW'
        else:
            prediction='Save Money! Your Family Will Love You '


        Age = request.form['age']
        Job = request.form['job']
        Marital = request.form['marital']
        Education = request.form['education']
        Default = request.form['default']
        Balance = request.form['balance']
        Housing = request.form['housing']
        Loan = request.form['loan']
        Contact = request.form['contact']
        Day = request.form['day']
        Month = request.form['month']
        Campaign = request.form['campaign']
        Pdays = request.form['pdays']
        Previous = request.form['previous']
        Poutcome = request.form['poutcome']

        customer1 = User(age=Age, job =Job, marital=Marital, education=Education, 
            default=Default, balance=Balance, housing=Housing, loan=Loan, 
            contact=Contact, day=Day, month=Month, campaign=Campaign, pdays=Pdays, 
            previous=Previous, poutcome=Poutcome, y =int(result))
        db.session.add(customer1)
        db.session.commit()


        return render_template("result.html",prediction=prediction, prob = result)




if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"], host=app.config["HOST"])
