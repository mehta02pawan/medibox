from sys import maxsize

# from importlib_metadata import metadata
import appt
# from flask_login import UserMixin
from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy import MetaData, Table, String, Column, Text, DateTime, Boolean, BigInteger, Integer
# from datetime import date
from flask import Blueprint
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

metadata = MetaData()

models = Blueprint('models', __name__)

Base = declarative_base()
# database_exists(engine.users)  
# create_database(engine.users)

class Users(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    f_name = Column(String(255))
    mobile = Column(BigInteger)
    email = Column(String(255))
    password = Column(String(255))
    birthday = Column(Integer)
    weight = Column(String(255))
    height = Column(String(255))
    created_at = Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __init__(self,f_name,mobile,email,password,birthday,weight,height, created_at):
        self.f_name = f_name
        self.email = email
        self.mobile = mobile
        self.password = password
        self.birthday = birthday
        self.weight = weight
        self.height = height
        self.created_at = created_at

class Prescriptions():
    id = Column(Integer, primary_key=True)
    dr_name = Column(String(255))
    dr_num = Column(String(255))
    address = Column(String(255))
    prescription_date = Column(DateTime , nullable = True)
    drugs = Column(String(255))
    u_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __init__(self,dr_name,dr_num,address,prescription_date,drugs,u_id, created_at):
        self.dr_name = dr_name
        self.dr_num = dr_num
        self.address = address
        self.prescription_date = prescription_date
        self.drugs = drugs
        self.u_id = u_id
        self.created_at = created_at

        
Prescriptions = Table('prescriptions', metadata,
    Column('id',Integer(), primary_key=True),
    Column('dr_name',String(255)),
    Column('dr_num',String(255)),
    Column('prescription_date',DateTime),
    Column('drugs',String(255)),
    Column('u_id',ForeignKey('users.id'))
)

def __init__(self,dr_name,dr_num,prescription_date,drugs,u_id):
    self.dr_name = dr_name
    self.dr_num = dr_num
    self.prescription_date = prescription_date
    self.drugs = drugs
    self.u_id = u_id

# Druglist = Table('drug_data3436', metadata,
#     Column('id',Integer(), primary_key=True),
#     Column('drugName',String(255)),
#     Column('condition',String(255))
# )

class Druglist(Base):
    __tablename__ = 'drug_data3436'
    id = Column(Integer, primary_key=True)
    drugName = Column(String(255))
    condition = Column(BigInteger)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

def __init__(self,drugName,condition, created_at):
    self.drugName = drugName
    self.condition = condition
    self.created_at = created_at
# prescriptions = relationship('Prescriptions', backref='user', lazy=True)
