# Example project repository

<!-- toc -->

- [Project Charter](#project-charter)
- [Project Backlog](#project-backlog)
- [Data Source](#data-source)
- [Repo structure](#repo-structure)
- [Documentation](#documentation)
- [Running the application](#running-the-application)
  * [1. Set up environment](#1-set-up-environment)
    + [With `virtualenv` and `pip`](#with-virtualenv-and-pip)
    + [With `conda`](#with-conda)
  * [2. Configure Flask app](#2-configure-flask-app)
  * [3. Initialize the database](#3-initialize-the-database)
  * [4. Run the application](#4-run-the-application)
- [Testing](#testing)

<!-- tocstop -->

## Project Charter 

**Vision**: This is a portrugess bank that spends excessive budget on marketing but does not have a satisfied result; there is not a significant change in the number of bank clients who buy the financial product (term deposit). In order to handle this problem, we need to evaluate the area of improvement for the bank's marketing strategy that can help ultimately increase more bank clients purchase the produce. In this way, the bank can earn more revenue

**Mission**: Drive subscription of the term deposit by using data of clients and marketing strategies to inform targeting decision 

**Success criteria**: 
1) Machine learning performance metric: Use a set of metrics (accuracy score, f1, AUC) for evaluation. The minimum value for accuracy is greater than 0.89 since the breakdown of the target variable is 89:11

2) Business outcome metric: The bank finds that there is an improvement in clients’ subscription rate of the term deposit 




## Project Backlog

**Theme 1**

* Development of the best model for identifying clients who are willing to purchase the financial product (term deposit) 

 **Epic** 
 
 * Comparison of all different models for prediction 
 
   **Backlog**
   
      * Perform summary analysis of each parameter and plot distribution –2 points (planned for the next two weeks)
   
   * Reduce the number of classes for categorical variables (month of contact) –2 points (planned for the next two weeks)
   
   * Convert continuous variables to categorical variables for convenience (age) –2 points (planned for the next two weeks)

   * Data preparation for different models using one hot encoding and other cleansing methods –2 points (planned for the next two weeks)

	* Detect any correlation between parameters and consider exclusion of those highly correlated –2 points (planned for the next two weeks)

     * Perform different methods to see variable importance and select those for later model training if necessary  –2 points (planned for the next two weeks)

    * Train different models that are applicable to the data set –4 points (planned for the next two weeks)
   
   * Interpret each model and select the most efficient and understandable one --2 points 

   * Examine which metric is more appropriate to evaluate all models –2 points 
      
       **Icebox**
   
    * Engineer features to describe clients’ affordability
    
    * Perform summary analysis of each parameter of new features generated and plot distribution 

    * Detect whether new features have strong correlation with features that are selected earlier 

**Theme 2**

* Classification of different clients’ types to help inform bank decision about advertising financial product to the right target population 

 **Epic** 
 
 * Analysis of different clusters of clients 
 
   **Backlog**
   
   * Select the proper number of clusters –2 points (planned for the next two weeks)
   
   * Compare clusters according to features used in segmentation –2 points (planned for the next two weeks)

	**Icebox**
 
   * Identify the relationship between clusters and clients’ decision of subscription of the term deposit 
  
  **Epic**
  * Deploy the model onto AWS and develop the subscription prediction App. Keep track of customer subscription in the bank as more customers are added.

	 **Backlog**
	 * Document bank clients' subscription rate and recalculate bank subscription each trial/quarter/year as new customers are added through the app on a continuous basis. The bank might have a certain goal as to what percentage of customers it wants to buy the financial product each year. If that is not met, then the bank has to evaluate its product, customer service, or marketing campaigns to increase retention.
	 *  Load the data in s3 
	 *  Set RDS for  accessing the data from the app 
   
## Data Source
*  https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

## Repo structure 

```
├── README.md                         <- You are here
│
├── app
├── static/                       <- CSS, JS files that remain static 
├──templates/                    <- HTML (or other code) that is templated and changes based on a set of inputs
│
├── config                            <- Directory for yaml configuration files for model training, scoring, etc
│   ├── logging/                      <- Configuration files for python loggers
│
├── data                              <- Folder that contains data used or generated. Only the external/ and sample/ subdirectories are tracked by git. 
│   ├── sample/                       <- Sample data used for code development and testing, will be synced with git
│
├── docs                              <- A default Sphinx project; see sphinx-doc.org for details.
│
├── models                            <- Trained model objects (TMOs), model predictions, and/or model summaries
│
├── notebooks
│   ├── develop                       <- Current notebooks being used in development.
│   ├── deliver                       <- Notebooks shared with others. 
│   ├── archive                       <- Develop notebooks no longer being used.
│
├── src                               <- Source data for the project 
│   ├── helpers/                      <- Helper scripts used in main src files 
│   ├── sql/                          <- SQL source code
│   ├── get_data.py                   <- Script for downloading data from the public s3 bucket
│   ├── upload_data.py                <- Script for uploading data files to s3 bucket. 
│   ├── load_data.py                  <- Script for loading data to the path specified 
│   ├── generate_features.py          <- Script for cleaning and transforming data and generating features used for use in training and scoring.
│   ├── train_model.py                <- Script for training machine learning model(s)
│   ├── score_model.py                <- Script for scoring new predictions using a trained model.
│   ├── evaluate_model.py             <- Script for evaluating model performance 
│   ├── model.py             	      <- Script for creating database model that is later connected to the Flask app.
│   ├── README.md                     <- Documentation with instructions to run scripts in src/ and midproject check.
│
├── test                              <- Files necessary for running model tests (see documentation below) 
│   ├── test.py                       <- Script for running unit tests on functions in src/.
├── run.py                            <- Simplifies the execution of one or more of the src scripts 
├── app.py                            <- Flask wrapper for running the model 
├── config.py                         <- Configuration file for Flask app
├── requirements.txt                  <- Python package dependencies 
```
This project structure was partially influenced by the [Cookiecutter Data Science project](https://drivendata.github.io/cookiecutter-data-science/).

## Documentation
 
* Open up `docs/build/html/index.html` to see Sphinx documentation docs. 
* See `docs/README.md` for keeping docs up to date with additions to the repository.

## Running the application 
### 1. Set up environment 

The `requirements.txt` file contains the packages required to run the model code. We need to change the path to the main repository before installing the requirement.txt. An environment can be set up in two ways. 

#### With `virtualenv`

```bash
pip install virtualenv

virtualenv pennylane

source pennylane/bin/activate

pip install -r requirements.txt

```
#### With `conda`

```bash
conda create -n avc python=3.7
conda activate avc
pip install -r requirements.txt

```

### 2. Configure Flask app 

`config.py` holds the configurations for the Flask app. It includes the following configurations, change the last line if run on RDS (more details in section 4):

```python
DEBUG = True  # Keep True for debugging, change to False when moving to production 
LOGGING_CONFIG = "config/logging/local.conf"  # Path to file that configures python logger
PORT = 3000  # What port to expose app on 
SQLALCHEMY_DATABASE_URI = 'sqlite:////data/user.db'  # URI for database that contains tracks

```


### 3. Initialize the database 

To create the database locally in the location configured in `config.py` with one initial bank customer, first change path to where the file is located and run: 

`cd path_to_repository/src`

`python model.py`


To create the database on RDS in the location configured in `config.py` with one initial bank customer, first change path to where the file is located and run: 

`cd path_to_repository/src`

`python model.py --RDS True`




### 4. Run the application 

To run the application locally, use the following code in config.py:

 `SQLALCHEMY_DATABASE_URI='sqlite:///data/user.db` 

To run the application on RDS, unncomment following chunk of code. Make sure you comment the above line because we already have the last line to find the database in RDS. 

```python
import os
conn_type = "mysql+pymysql"
user = os.environ.get("MYSQL_USER")
password = os.environ.get("MYSQL_PASSWORD")
host = os.environ.get("MYSQL_HOST")
port = os.environ.get("MYSQL_PORT")
DATABASE_NAME = 'msia423'
SQLALCHEMY_DATABASE_URI = "{}://{}:{}@{}:{}/{}".format(conn_type, user, password, host, port, DATABASE_NAME)
```

Change the existing code to the following: 
```python
HOST = '0.0.0.0'
```

After adopting corresponding changes, run

 ```bash
 cd path_to_repository
 python app.py 

 ```

### 5. Interact with the application 

Go to [http://127.0.0.1:3000/]( http://127.0.0.1:3000/) to interact with the current version of the app. Please click the discover button on the first image to make the prediction.

## Testing 

Run `make test` from the command line in the main project repository. 


Tests exist in `test/test.py`
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzOTQyMzExOTRdfQ==
-->

