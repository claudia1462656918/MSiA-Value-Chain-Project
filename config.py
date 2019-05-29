import os
DEBUG = True
LOGGING_CONFIG = "config/logging/local.conf"
PORT = 3000
APP_NAME = "penny-lane"
conn_type = "mysql+pymysql"
user = os.environ.get("MYSQL_USER")
password = os.environ.get("MYSQL_PASSWORD")
host = os.environ.get("MYSQL_HOST")
port = os.environ.get("MYSQL_PORT")
DATABASE_NAME = 'msia423'
SQLALCHEMY_DATABASE_URI = "{}://{}:{}@{}:{}/{}".format(conn_type, user, password, host, port, DATABASE_NAME)
#SQLALCHEMY_DATABASE_URI = 'sqlite:///data/user.db'
SQLALCHEMY_TRACK_MODIFICATIONS = True
HOST = "0.0.0.0"
MAX_ROWS_SHOW = 20
