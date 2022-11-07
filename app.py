# from flask_mysqldb import MySQL
from datetime import datetime
from fileinput import filename
import create_event
from collections import Counter
from flask import Flask, render_template, request, send_file, redirect, url_for, session, flash, jsonify
from json import dump
import requests
from isodate import parse_duration
# import matplotlib.pyplot as plt
# import flask_mysqldb
from pickle import GET
from flask_sqlalchemy import SQLAlchemy
from pyparsing import autoname_elements
from sqlalchemy import func
import pytesseract
import numpy as np
import pandas as pd
import re
from word2number import w2n
import cv2
from pprint import pprint
from dateparser.search import search_dates
from dateparser import parser
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
import sqlalchemy as db
from sqlalchemy import exists
# from sqlalchemy.testing.suite.test_reflection import users
from sqlalchemy.orm import sessionmaker
import models
from models import Prescriptions, Users, Uploads, Drugdata, bmi
from werkzeug.utils import secure_filename
import os
import json
from settings import app, db, engine, drugname
import spacy
import geocoder
import chat
import json
from flask_socketio import SocketIO
import logging

# app.config['MYSQL_HOST'] = '127.0.0.1'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = ''
# app.config['MYSQL_DB'] = 'trial'
#
# app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
# mysql = flask_mysqldb.MySQL(app)
socketio = SocketIO(app)

connection = engine.connect()
metadata = db.MetaData()

Session = sessionmaker(bind=engine)
session = Session()
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"


@login_manager.user_loader
def load_user(user_id):
    return Users.query.get(int(user_id))


@app.route('/layout')
def layout():
    return render_template('layout.html')


@app.route('/registration', methods=['GET', 'POST'])
def registration():
    if request.method == 'POST':
        email = request.form.get('email')
        # prescription = db.Table('users', metadata, autoload=True, autoload_with=engine)
        user = db.session.query(db.exists().where(Users.email == email)).scalar()
        # user = db.select([prescription]).where(prescription.columns.email == email)
        # found_user = Users.query.filter_by(email = email).first()
        if user:
            flash("Email already exists!!!!", 'danger')
            return redirect(url_for('login'))
        else:
            email = request.form.get('email')
            name = request.form.get('name')
            password = request.form.get('password')
            birthday = request.form.get('dob')
            weight = request.form.get('weight')
            height = request.form.get('height')
            mobile = request.form.get('mobile')
            usr = Users(name, mobile, email, password, birthday, weight, height)
            db.session.add(usr)
            db.session.commit()
            # session['id'] = id

            flash('Registered Successfully!', 'info')
            return redirect(url_for('home'))
    return render_template('registration.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    username = request.form.get('email')
    pass1 = request.form.get('password')
    user = Users.query.filter_by(email=username).first()
    if username is not None:
        if user:
            found_user = Users.query.filter_by(email=username, password=pass1).first()
            if found_user:
                login_user(user)
                # session.delete(user)
                session.permanent = False
                # local_object = db.session.merge(current_user)
                # db.session.add(local_object)
                # db.session.commit()
                # db.session.add(current_user)
                #
                # db.session.commit()

                return redirect(url_for('home'))
            else:
                flash("Wrong Password", "danger")
                return redirect(url_for('login'))
        else:
            flash("User doesn't exist!!!")
            return render_template("login.html")
    return render_template("login.html")


@app.route('/logout', methods=['GET', 'POST'])
@login_required
@socketio.on('disconnect')
def logout():
    # session.clear()
    logout_user()
    flash("See you again!")
    # Session.pop('health1234', None)
    return redirect(url_for('login'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    image = request.files['image']
    diagnosis = request.form.get('diagnosis')
    if not image:
        return 'no image uploaded', 400
    filename = secure_filename(image.filename)
    mimetype = image.mimetype

    path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(path)
    img = Uploads(image_name=filename, mimetype=mimetype,
                  diagnosis=diagnosis, user_id=current_user.id)

    db.session.add(img)
    db.session.commit()
    img_1 = cv2.imread(path)
    gray_image = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    dilation = cv2.dilate(threshold_img, rect_kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Creating a copy of image
    im2 = img_1.copy()

    # A text file is created and flushed
    file = open("recognized.txt", "w+")
    file.write("")
    file.close()

    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Drawing a rectangle on copied image
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Cropping the text block for giving input to OCR
        cropped = im2[y:y + h, x:x + w]

        # Open the file in append mode
        file = open("recognized.txt", "w")

        # Apply OCR on the cropped image
        extractedInformation = pytesseract.image_to_string(threshold_img)

    # extractedInformation = pytesseract.image_to_string(threshold_img)
    # print(extractedInformation)
    final_list = extractedInformation.split('\n')
    prediction = map(extract_drug_entity, final_list)
    numbersSquare = list(prediction)
    list2 = [x for x in numbersSquare if x]
    dr = [tup[0] for tup in list2]
    dr1 = [tup[0] for tup in dr]
    dr2 = [str(x).lower() for x in dr1]
    pattern = re.findall(r'(?P<duration>\w+)\s+\w+\s+daily [\w+][\s+](?P<num_days>\d+)', extractedInformation)
    # (duration, days)
    desc = re.findall('\w+\s+\w+\s+daily [\w+][\s+]\d+', extractedInformation)
    print(desc)
    drug_dose = [(2, '1'), (4, '7')]
    # [(w2n.word_to_num(elem[0]), elem[1]) for elem in pattern]
    extractedInformation = extractedInformation.strip()
    extractedInformation = re.sub('\\s+Or. | \\s+Dr. ', ' Dr. ', extractedInformation)
    dr_name = re.split('Dr. ', extractedInformation)[1].split('\n')[0]
    dr_num_pattern = re.compile('(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})(?: *x(\d+))?')
    dr_num = dr_num_pattern.search(extractedInformation)
    dr_num = dr_num.group(0)
    dates = search_dates(extractedInformation, settings={'STRICT_PARSING': True})
    dates_df = pd.DataFrame(dates)
    filter = dates_df[0].str.contains('\d{5}')
    filter_dates = dates_df[~filter]
    sorted_dates = list(filter_dates[1].sort_values())
    # birth_date = sorted_dates[0]
    prescription_date = sorted_dates[1]
    drg = list(set(dr2).intersection(drugname))

    create_event.loop(drug_dose, prescription_date, drg, desc)
    drugs = np.array(drg)

    # address
    p1 = re.compile('\d{1,4}[ ][A-Z]')
    start = p1.search(extractedInformation).span()[0]
    p2 = re.compile('[A-Za-z]\d[A-Za-z][ ]?\d[A-Za-z]\d')
    end = p2.search(extractedInformation).span()[1]
    zipcode = p2.search(extractedInformation).group(0)
    addr = extractedInformation[start:end]
    address = ' '.join(i for i in addr.split() if "\\" not in i)

    pres_data = Prescriptions(dr_name=dr_name, dr_num=dr_num, prescription_date=prescription_date, drugs=drugs,
                              address=address, zipcode=zipcode,
                              upload_id=img.id)
    db.session.add(pres_data)
    db.session.commit()

    return redirect(url_for('home'))


nlp2 = spacy.load("D:\\majorproject\\ner_new")


def extract_drug_entity(text):
    docx = nlp2(text)
    result = [(ent, ent.label_) for ent in docx.ents]
    return result


@app.route('/', methods=['GET', 'POST'])
@login_required
def home():
    engine = db.create_engine('mysql://root@127.0.0.1/health', {})
    connection = engine.connect()
    u_id = current_user.id
    prescription = db.Table('prescriptions', metadata, autoload=True, autoload_with=engine)
    upload = db.Table('uploads', metadata, autoload=True, autoload_with=engine)

    u_name = current_user.name
    search = request.form.get('search')
    if search:
       dcount = db.session.query(db.func.count(prescription.columns.dr_name), prescription.columns.dr_name,
                              prescription.columns.dr_num).where(prescription.columns.upload_id.in_(
        db.select([upload.columns.id]).where(upload.columns.user_id == u_id))).group_by(
        prescription.columns.dr_name, prescription.columns.drugs).having(db.func.count(prescription.columns.dr_name) >= 1).filter(db.or_(prescription.columns.dr_name.contains(search), prescription.columns.drugs.contains(search))).all()

    else:
        dcount = db.session.query(db.func.count(prescription.columns.dr_name), prescription.columns.dr_name,
                              prescription.columns.dr_num).where(prescription.columns.upload_id.in_(
        db.select([upload.columns.id]).where(upload.columns.user_id == u_id))).group_by(
        prescription.columns.dr_name).having(db.func.count(prescription.columns.dr_name) >= 1).all()

    #graph data
    dr_name_gr = [dr[1].split(" ") for dr in dcount]
    cnt_gr = [int(dr[0]) for dr in dcount]
    tot_visiits = sum(cnt_gr)
    pr_data = db.session.query(prescription.columns.prescription_date).where(prescription.columns.upload_id.in_(
        db.select([upload.columns.id]).where(upload.columns.user_id == u_id))).where(db.func.year(prescription.columns.prescription_date) == datetime.now().year).all()
    pr_date = [str(g[0].strftime('%B')) for g in pr_data]
    # print(dr_name_gr)
    pr_month = Counter(pr_date).keys()
    pr_count = Counter(pr_date).values()
    print(pr_month,pr_count)


    # bmi graph data
    bmi_table = db.Table('bmi', metadata, autoload=True, autoload_with=engine)
    query1 = db.select([bmi_table.columns.bmi, bmi_table.columns.created_at, bmi_table.columns.weight]).where(
    bmi_table.columns.user_id == current_user.id)
    graph = connection.execute(query1).fetchall()
    bmi_gr = [float(g[0]) for g in graph]
    avg_bmi = round(sum(bmi_gr)/len(bmi_gr),2)
    cr_weight = [g[2] for g in graph][-1]
    created_at_gr = [str(g[1]).split(" ")[0] for g in graph]
    # created_at_gr = [str(g[1].strftime('%B')) for g in graph]

    # total number of drugs
    drug_cnt = connection.execute(db.select(prescription.columns.drugs))
    cnt = drug_cnt.fetchall()
    count = 0
    for data in cnt:
        count += len(data)



    return render_template("home.html", query1=dcount, name=u_name, dr_name_gr=dr_name_gr, cnt_gr=cnt_gr, bmi_gr=bmi_gr[-6:],
                           created_at_gr=created_at_gr[-6:], count=count, pr_count=list(pr_count),pr_month=list(pr_month), cr_weight=cr_weight, avg_bmi=avg_bmi, tot_visiits=tot_visiits)


@app.route('/<dr_name>', methods=['GET', 'POST'])
def dr_name(dr_name):
    # userData = Users.query.filter_by(id=id).first()
    prescription = db.Table('prescriptions', metadata, autoload=True, autoload_with=engine)
    upload = db.Table('uploads', metadata, autoload=True, autoload_with=engine)

    dcount1 = db.session.query(
        upload, prescription).filter(prescription.columns.dr_name == dr_name).filter(
        prescription.columns.upload_id == upload.columns.id).filter(upload.columns.user_id == current_user.id).all()

    return render_template('prescription.html', userData=current_user, drName=dcount1)


@app.route('/profile/<int:id>', methods=['GET', 'POST'])
def profile(id):
    engine = db.create_engine('mysql://root@127.0.0.1/health', {})
    connection = engine.connect()
    census = db.Table('users', metadata, autoload=True, autoload_with=engine)
    update_user = Users.query.get_or_404(id)
    pas = current_user.password
    print(pas)
    gender = current_user.gender
    if gender == "male":
        avatar = 'images/male.png'
    else:
        avatar = 'images/female.png'

    if request.method == 'POST':
        update_user.name = request.form.get('name')
        # print(update_user.name)
        update_user.email = request.form.get('email')
        update_user.password = request.form.get('new_password')
        if update_user.password == '':
            update_user.password = pas
        update_user.birthday = request.form.get('birthday')
        update_user.gender = request.form.get('gender')
        update_user.weight = request.form.get('weight')
        update_user.height = request.form.get('height')
        update_user.mobile = request.form.get('mobile')

        updated = db.update(census).values(name=update_user.name, email=update_user.email,
                                           password=update_user.password, mobile=update_user.mobile,
                                           birthday=update_user.birthday, gender=update_user.gender,
                                           height=update_user.height, weight=update_user.weight)
        query = updated.where(census.columns.id == id)
        results = connection.execute(query)

        # up_user = Users.query.filter_by(id).update(update_user.name,  update_user.mobile, update_user.email, update_user.password,update_user.birthday, gender, update_user.weight, update_user.height)
        db.session.commit()

    return render_template("profile_page.html", avatar=avatar)


@app.route('/base')
def base():
    return render_template("chat1.html")


@app.route('/chat', methods=['GET', 'POST'])
def chatbotResponse():
    if request.method == 'POST':
        the_question = request.get_json().get("message")
        response = chat.chatbot_response(the_question)
        message = {"answer": response}

    return jsonify(message)


# @app.route('/bmi', methods=['GET', 'POST'])
# def bmi():
#     # h = int(current_user.height)
#     # w = int(current_user.weight)
#     # bmi = round((w/(h**2))*10000,2)
#     return render_template("bmi1.html")

@app.route('/bmi1', methods=['GET', 'POST'])
def bmi1():
    h = int(current_user.height)
    w = int(current_user.weight)
    bmi_info = round((w / (h ** 2)) * 10000, 2)
    return render_template("bmi2.html", bmi=bmi_info)


@app.route('/bmi', methods=['GET', 'POST'])
def bmi():
    engine = db.create_engine('mysql://root@127.0.0.1/health', {})
    connection = engine.connect()

    if request.method == 'POST':
        census = db.Table('users', metadata, autoload=True, autoload_with=engine)
        bmi_table = db.Table('bmi', metadata, autoload=True, autoload_with=engine)
        # update_user = Users.query.get_or_404(id)
        weight = int(request.form.get('weight'))
        height = int(request.form.get('height'))

        bmi_info = round((weight / (height ** 2)) * 10000, 2)

        updated = db.update(census).values(height=height, weight=weight)
        bmi_data = connection.execute(
            db.insert(bmi_table).values(height=height, weight=weight, bmi=bmi_info, user_id=current_user.id))
        query = updated.where(census.columns.id == current_user.id)
        results = connection.execute(query)
        db.session.commit()
    bmi_table = db.Table('bmi', metadata, autoload=True, autoload_with=engine)
    query1 = db.select([bmi_table.columns.bmi, bmi_table.columns.created_at]).where(
        bmi_table.columns.user_id == current_user.id)
    graph = connection.execute(query1).fetchall()
    bmi_gr = [g[0] for g in graph]
    created_at_gr = [str(g[1]).split(" ")[0] for g in graph]
    if len(graph) > 6:
        bmi_gr = bmi_gr[-6:]
        created_at_gr = created_at_gr[-6:]

    return render_template('bmi1.html', bmi_gr=bmi_gr, created_at_gr=created_at_gr)


@app.route('/maps', methods=['GET', 'POST'])
def maps():
    g = geocoder.ip('me')
    loc = g.latlng
    url = "https://google-maps28.p.rapidapi.com/maps/api/place/nearbysearch/json"

    querystring = {"location": ','.join([str(n) for n in loc]),
                   "radius": "5000",
                   "language": "en",
                   "type": ["doctor", "hospital"]
                   }

    headers = {
        "X-RapidAPI-Key": "e182587402msh8fb9cb7826edf43p1eda64jsnb099563a5477",
        "X-RapidAPI-Host": "google-maps28.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    data = response.json()
    print(data)

    print(g.latlng)
    return render_template('maps.html', data=data, loc=loc)


@app.route('/videos', methods=['GET', 'POST'])
def videos():
    h = int(current_user.height)
    w = int(current_user.weight)
    bmi_data = round((w / (h ** 2)) * 10000, 2)
    if (bmi_data < 18.5):
        data = "underweight"

    elif (bmi_data > 18.5 and bmi_data < 24.9):
        data = "healthy"
    elif (bmi_data > 25 and bmi_data < 29.9):
        data = "overweight"
    else:
        data = "obese"

    search_url = 'https://www.googleapis.com/youtube/v3/search'
    videos_url = 'https://www.googleapis.com/youtube/v3/videos'
    params = {
        'key': 'AIzaSyChdx2e082EBpKVzC_NudAI9z6tpakgjSA',
        'q': 'workout for' + data + 'people',
        'part': 'snippet',
        'maxResults': 6,
        'type': 'video'
    }

    r = requests.get(search_url, params=params)
    results = r.json()['items']
    video_ids = []
    videos = []
    for result in results:
        video_ids.append(result['id']['videoId'])

    video_params = {
        'key': 'AIzaSyChdx2e082EBpKVzC_NudAI9z6tpakgjSA',
        'id': ','.join(video_ids),
        'part': 'snippet,contentDetails',
        'maxResults': 6

    }
    r = requests.get(videos_url, params=video_params)
    results = r.json()['items']
    for result in results:
        video_data = {
            'id': result['id'],
            'url': f'https://www.youtube.com/watch?v={result["id"]}',
            'thumbnail': result['snippet']['thumbnails']['high']['url'],
            'duration': int(parse_duration(result['contentDetails']['duration']).total_seconds() // 60),
            'title': result['snippet']['title'],
        }
        videos.append(video_data)
    return render_template('videos.html', videos=videos)


@app.route('/scheduler')
def scheduler():
    return render_template('calendar.html')


if __name__ == '__main__':
    app.run(debug=True)
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.ERROR)
