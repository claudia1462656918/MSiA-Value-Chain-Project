import os
import sys
import logging
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy as sql

import argparse


logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
logger = logging.getLogger('sql_db')


Base = declarative_base()


# ADD CLASS FOR  USER TABLE HERE
class User(Base):
    """ Defines the data model for the table `tweetscore`. """

    __tablename__ = 'user'

    id = Column(Integer, primary_key=True, unique=True, nullable=False)
    age = Column(Integer, unique=False, nullable=False)
    job = Column(Integer, unique=False, nullable=False)
    marital = Column(Integer, unique=False, nullable=False)
    education = Column(Integer, unique=False, nullable=False)
    default = Column(Integer, unique=False, nullable=False)
    balance = Column(Integer, unique=False, nullable=False)
    housing = Column(Integer, unique=False, nullable=False)
    loan = Column(Integer, unique=False, nullable=False)
    contact = Column(Integer, unique=False, nullable=False)
    day = Column(Integer, unique=False, nullable=False)
    month = Column(Integer, unique=False, nullable=False)
    campaign = Column(Integer, unique=False, nullable=False)
    pdays = Column(Integer, unique=False, nullable=False)
    previous = Column(Integer, unique=False, nullable=False)
    poutcome = Column(Integer, unique=False, nullable=False)
    y = Column(Integer, unique=False, nullable=False)
    # user_input = Column(Integer, unique=False, nullable=False)
    # user_output = Column(Integer, unique=False, nullable=False)
    # user_feedback = Column(String(100), unique=False, nullable=False)
    

    # def __repr__(self):
    #     user_repr = "<User(ip_time='%s', age = '%d', job = '%d', marital='%d',education='%d',default='%d',balance='%d',housing='%d',loan='%d',contact='%d',day='%d',month='%d',campaign='%d',pdays='%d',previous='%d',poutcome='%d',y='%d')>"
    #     return user_repr % (self.ip_time, self.age, self.job, self.marital, self.education, self.default, self.balance,
    #         self.housing, self.loan, self.contact, self.day, self.month, self.campaign, self.pdays, self.previous,
    #         self.poutcome, self.y)




def get_engine_string(RDS = False):
    if RDS:
        conn_type = "mysql+pymysql"
        user = os.environ.get("MYSQL_USER")
        password = os.environ.get("MYSQL_PASSWORD")
        host = os.environ.get("MYSQL_HOST")
        port = os.environ.get("MYSQL_PORT")
        DATABASE_NAME = 'msia423'
        engine_string = "{}://{}:{}@{}:{}/{}". \
            format(conn_type, user, password, host, port, DATABASE_NAME)
        # print(engine_string)
        logging.debug("engine string: %s"%engine_string)
        return  engine_string
    else:
        return 'sqlite:///user.db' # relative path




def create_db(args, engine=None, engine_string=None):
    """Creates a database with the data models inherited from `Base` (Tweet and TweetScore).
    Args:
        engine (:py:class:`sqlalchemy.engine.Engine`, default None): SQLAlchemy connection engine.
            If None, `engine_string` must be provided.
        engine_string (`str`, default None): String defining SQLAlchemy connection URI in the form of
            `dialect+driver://username:password@host:port/database`. If None, `engine` must be provided.
    Returns:
        None
    """


    if engine is None:
        RDS = eval(args.RDS) # evaluate string to bool
        logger.info("RDS:%s"%RDS)
        engine = sql.create_engine(get_engine_string(RDS = RDS))

    Base.metadata.create_all(engine)
    logging.info("database created")
    return engine
    # create a db session
    # Session = sessionmaker(bind=engine)  
    # session = Session()




    # use1 = User(age=0, job =1, marital=0, education=1, default=0, balance=2000, 
    # housing=0, loan=0, contact=1, day=12, month=0, campaign=2, pdays=2, previous=2, poutcome=0, y =1)
    # session.add(use1)
    # session.commit()

    # logger.info("Data added")

    # query = "SELECT * FROM user"
    # df = pd.read_sql(query, con=engine)
    # logger.info(df)
    # session.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create defined tables in database")
    parser.add_argument("--RDS", default="False",help="True if want to create in RDS else None")
    args = parser.parse_args()
    engine = create_db(args)

    Session = sessionmaker(bind=engine)  
    session = Session()

    use1 = User(age=0, job =1, marital=0, education=1, default=0, balance=2000, 
    housing=0, loan=0, contact=1, day=12, month=0, campaign=2, pdays=2, previous=2, poutcome=0, y =1)
    session.add(use1)
    session.commit()


    logger.info("Data added")

    query = "SELECT * FROM user"
    df = pd.read_sql(query, con=engine)
    logger.info(df)
    session.close()