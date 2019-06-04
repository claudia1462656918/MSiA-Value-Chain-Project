DEBUG = True
LOGGING_CONFIG = "config/logging/local.conf"
PORT = 3000
APP_NAME = "bank-customer-prediction"
# import os
# conn_type = "mysql+pymysql"
# user = os.environ.get("MYSQL_USER")
# password = os.environ.get("MYSQL_PASSWORD")
# host = os.environ.get("MYSQL_HOST")
# port = os.environ.get("MYSQL_PORT")
# DATABASE_NAME = 'msia423'
# SQLALCHEMY_DATABASE_URI = "{}://{}:{}@{}:{}/{}".format(conn_type, user, password, host, port, DATABASE_NAME)
SQLALCHEMY_DATABASE_URI = 'sqlite:///data/user.db'
SQLALCHEMY_TRACK_MODIFICATIONS = True
HOST = "127.0.0.1"
MODEL_PATH = "models/bank-prediction.pkl"
