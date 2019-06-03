import argparse
import logging.config
import yaml
import os

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, MetaData, Float
from sqlalchemy.orm import sessionmaker



logger = logging.getLogger(__name__)
logger.setLevel("INFO")

Base = declarative_base()


# ADD CLASS FOR  USER TABLE HERE
class User(Base):
    """ Defines the data model for the table `tweetscore`. """

    __tablename__ = 'user'

    ip_time = Column(Integer, primary_key=True, unique=True, nullable=False)
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
    y = Column(String(100), unique=False, nullable=False)
    
    def __repr__(self):
        return '<User %r>' % self.ip_time