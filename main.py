import base64
import os
import pickle

from flask import Flask, redirect, url_for, render_template, request, flash, session
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
from datetime import datetime
import operator
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import glob
from skimage.transform import resize
from ContourWithData import ContourWithData
from sklearn.metrics import classification_report

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'db_plat_app'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
app.config['SECRET_KEY'] = 'thisisasecret'
app.permanent_session_lifetime = timedelta(minutes=5)
categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
              'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
model = pickle.load(open(r'C:\Users\ACER\PycharmProjects\FlaskTry\static\model_knn_3.p', 'rb'))
list_plate_dir = ["www"]

mysql = MySQL(app)
MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
@app.route('/tryyy')
def tryyy():
    return render_template('trying.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        admin = "admin_parkir"
        password = "11223344"
        session.permanent = True  # <--- makes the permanent session
        pswd = request.form["pass"]
        user = request.form["nm"]
        if user != admin:
            return render_template('wrong.html')
        elif pswd != password:
            return render_template('wrong.html')
        else:
            session["user"] = user
            return redirect(url_for("all_data"))
    else:
        if "user" in session:
            return redirect(url_for("all_data"))

        return render_template("login.html")

@app.route('/db', methods=['POST', 'GET'])
def db():
    if request.method == 'POST':
        id = 0
        nama = request.form['nama']
        alamat = request.form['alamat']
        no_hp = request.form['no_hp']
        plat = request.form['plat']

        cursor = mysql.connection.cursor()
        cursor.execute('''INSERT INTO tb_identifikasi (id, nama, alamat, no_hp, plat)  VALUES (%s,%s,%s,%s,%s)''', (id, nama, alamat, no_hp, plat))
        #cursor.execute('''INSERT INTO db_coba (id, nama) VALUES(%s, %s)''',(id, nama))

        mysql.connection.commit()
        cursor.close()
        return redirect(url_for('all_data'))

@app.route("/user")
def user():
    if "user" in session:
        user = session["user"]
        return render_template('all_data.html', user = user)
    else:
        return redirect(url_for("login"))

@app.route('/all_data')
def all_data():
    con = mysql.connection
    cursor = con.cursor()
    query ="SELECT * FROM tb_identifikasi"
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    return render_template('all_data.html', all_data = rows)

@app.route('/tambah_data')
def tambah_data():
    return render_template('data_pemilik.html')

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/capture_img', methods=['POST'])
def capture_img():
    if request.method == 'POST':
        path = r"static/img/camera//"
        for file_name in os.listdir(path):
            # construct full file path
            file = path + file_name
            if os.path.isfile(file):
                os.remove(file)

        image = request.form['mydata']
        img_binary = base64.b64decode(image)
        img_jpg = np.frombuffer(img_binary, dtype=np.uint8)
        img = cv2.imdecode(img_jpg, cv2.IMREAD_ANYCOLOR)
        dim = (756, 567)
        resized_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        now = datetime.now()
        current_time = now.strftime("%d_%m_%Y_%H_%M_%S")

        image_file = "static/img/camera/gambarplat_%s.jpg" % current_time
        cv2.imwrite(image_file, resized_image)

    return render_template('makesure.html', img=image_file)


@app.route('/process', methods=['POST', 'GET'])
def process():
    if request.method == 'POST':
        path = r"static/img/cropped//"
        for file_name in os.listdir(path):
            # construct full file path
            file = path + file_name
            if os.path.isfile(file):
                os.remove(file)

        path_char = r"static/img/char//"
        for file_name in os.listdir(path_char):
            # construct full file path
            file = path_char + file_name
            if os.path.isfile(file):
                os.remove(file)

        # Import Haar Cascade XML file
        haar_cascade = cv2.CascadeClassifier('static/cascade2.xml')

        # Read car image and convert color to RGB

        plate_img_norgbb = cv2.imread("static/img/fix/plat 24.JPG")

        #image = request.form['imgg']
        plate_img_norgbb = cv2.imread(image)

        width = int(plate_img_norgbb.shape[1] * 2)
        height = int(plate_img_norgbb.shape[0] * 2)
        dim = (width, height)
        resized_image = cv2.resize(plate_img_norgbb, dim, interpolation=cv2.INTER_AREA)

        plate_overlays = resized_image.copy()
        plate_overlays2 = resized_image.copy()  # Create overlay to display red rectangle of detected car plate
        plate_rects = []
        plate_rects = haar_cascade.detectMultiScale(plate_overlays, scaleFactor=1.1, minNeighbors=5)

        yo = all(map(lambda x: x is None, plate_rects))

        if yo == True:
            return render_template('undetected.html')

        else:
            x, y, w, h = plate_rects[0]

            now = datetime.now()
            current_time = now.strftime("%H%M%S")

            list_plate_dir.pop(0)
            plate_dir = "static/img/cropped/plate_%s.jpg" % current_time
            list_plate_dir.append(plate_dir)

            cv2.putText(plate_overlays, 'Plat Nomer', (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
            cv2.rectangle(plate_overlays, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imwrite(plate_dir, plate_overlays)

            plate_cropped_dir = "static/img/cropped/plate_cropped_%s.jpg" % current_time
            carplate_img = plate_overlays2[y: y + h, x: x + w]
            width = (int(carplate_img.shape[1]) * 2)
            height = (int(carplate_img.shape[0]) * 2)
            dim = (width, height)
            resized_image_plate = cv2.resize(carplate_img, dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite(plate_cropped_dir, resized_image_plate)

            result = resized_image_plate

            # convert to grayscale and blur the image
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)

            # Applied inversed thresh_binary
            binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

            cont, _ = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # creat a copy version "test_roi" of plat_image to draw bounding box
            test_roi = result.copy()

            # Initialize a list which will be used to append charater image
            crop_characters = []

            # define standard width and height of character
            digit_w, digit_h = 60, 110

            for c in sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h / w
                if 1 <= ratio <= 3.5:  # Only select contour with defined ratio
                    if h / result.shape[0] >= 0.3:  # Select contour which has the height larger than 30% of the plate
                        # Draw bounding box arroung digit number
                        w = w + 10
                        h = h + 10

                        cv2.rectangle(test_roi, (x - 5, y - 5), (x + w, y + h), (0, 255, 0), 2)
                        # print(x,y,w,h)

                        # Sperate number and gibe prediction
                        curr_num = gray[y:y + h, x:x + w]
                        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                        # _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        crop_characters.append(curr_num)

            if len(crop_characters) <= 0:
                return render_template('undetected.html', image=image)

            else:
                for i in range(len(crop_characters)):
                    cv2.imwrite(os.path.join(path_char, str(i) + '_' + current_time + '.jpg'), crop_characters[i])

                return render_template('process.html', plate_dir=plate_dir, plate_cropped_dir=plate_cropped_dir)


@app.route('/char', methods=['POST', 'GET'])
def char():
    files = []
    path_char = r"static/img/char//"

    for file_name in os.listdir(path_char):
        # construct full file path
        file = path_char + file_name
        if os.path.isfile(file):
            files.append(file)

    files = sorted(files)
    print(files)
    i = 0
    chars = []
    for file in files:
        img = cv2.imread(file, 0)
        blur = cv2.GaussianBlur(img, (7, 7), 0)
        binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img_resize = resize(binary, (25, 25, 3))
        l = [img_resize.flatten()]
        probability = model.predict_proba(l)
        char = categories[model.predict(l)[0]]
        chars.append(char)
        print(categories[model.predict(l)[0]], end=' ')

    for i in chars:

        if chars[0] == 'q':
            chars[0] = 'g'
        elif chars[0] == '0':
            chars[0] = 'g'
        elif chars[0] == '6':
            chars[0] = 'g'


        elif chars[1] == 'q':
            chars[1] = '0'
        elif chars[1] == 'b':
            chars[1] = '8'
        elif chars[1] == 'z':
            chars[1] = '2'
        elif chars[1] == 'e':
            chars[1] = '6'
        elif chars[1] == 's':
            chars[1] = '3'
        elif chars[1] == 't':
            chars[1] = '7'

        elif chars[2] == 'g':
            chars[2] = '6'
        elif chars[2] == 'q':
            chars[2] = '0'
        elif chars[2] == 'b':
            chars[2] = '8'
        elif chars[2] == 'z':
            chars[2] = '2'
        elif chars[2] == 'e':
            chars[2] = '6'
        elif chars[2] == 's':
            chars[2] = '3'
        elif chars[2] == 't':
            chars[2] = '7'

        elif chars[3] == 'g':
            chars[3] = '6'
        elif chars[3] == 'q':
            chars[3] = '0'
        elif chars[3] == 'b':
            chars[3] = '8'
        elif chars[3] == 'z':
            chars[3] = '2'
        elif chars[3] == 'e':
            chars[3] = '6'
        elif chars[3] == 's':
            chars[3] = '3'
        elif chars[3] == 't':
            chars[3] = '7'

        elif chars[-1] == '6':
            chars[-1] = 'g'
        elif chars[-1] == '0':
            chars[-1] = 'q'
        elif chars[-1] == '8':
            chars[-1] = 'b'
    plate_img = list_plate_dir[0]
    comma_separated = ''.join(chars)
    print(comma_separated)
    print(plate_img)
    return render_template("read.html", chars=comma_separated, plate_img=plate_img)

@app.route('/report')
def report():
       con = mysql.connection
       cursor = con.cursor()
       queryss = "SELECT * FROM tb_report"
       cursor.execute(queryss)
       report = cursor.fetchall()
       cursor.close()
       return render_template('report.html', report = report)

@app.route('/pemilik', methods = ['POST', 'GET'])
def pemilik():
    if request.method == "POST":
        char_plate = request.form["karakter"]
        con = mysql.connection
        cursor = con.cursor()
        querys = "SELECT * FROM tb_identifikasi WHERE plat = %s"
        pl = (char_plate,)
        plate_img = list_plate_dir[0]
        print(plate_img)
        cursor.execute(querys, pl)
        rowss = cursor.fetchall()
        cursor.close()
        if len(rowss)>0:
            return render_template('pemilik.html', plat=rowss, plate_img = plate_img)
        else:
            return render_template('pemilik_baru_auto.html',char_plate = char_plate)

@app.route('/store_info', methods = ['POST', 'GET'])
def store_info():
    if request.method == 'POST':
        id = 0
        plat = request.form['plat_report']
        pemilik = request.form['pemilik_report']
        time_report = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cursor = mysql.connection.cursor()
        cursor.execute('''INSERT INTO tb_report (id, plat, pemilik, time_report)  VALUES (%s,%s,%s,%s)''', (id, plat, pemilik, time_report))

        mysql.connection.commit()
        cursor.close()
        return redirect(url_for('index'))

@app.route('/db_auto', methods=['POST', 'GET'])
def db_auto():
    if request.method == 'POST':
        id = 0
        nama = request.form['nama_auto']
        alamat = request.form['alamat_auto']
        no_hp = request.form['no_hp_auto']
        plat = request.form['plat_auto']
        time_report = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cursor = mysql.connection.cursor()
        cursor.execute('''INSERT INTO tb_identifikasi (id, nama, alamat, no_hp, plat)  VALUES (%s,%s,%s,%s,%s)''', (id, nama, alamat, no_hp, plat))
        cursor.execute('''INSERT INTO tb_report (id, plat, pemilik, time_report)  VALUES (%s,%s,%s,%s)''', (id, plat, nama, time_report))

        mysql.connection.commit()
        cursor.close()
        return redirect(url_for('index'))

@app.route('/delete/<id>')
def delete(id):
    cur = mysql.connection.cursor()
    cur.execute('DELETE FROM tb_identifikasi WHERE id=%s', (id,))
    mysql.connection.commit()
    return redirect(url_for('all_data'))

@app.route('/edit/<id>',  methods=['POST', 'GET'])
def edit(id):
    cur = mysql.connection.cursor()
    cur.execute('SELECT * FROM tb_identifikasi WHERE id=%s', (id,))
    result = cur.fetchall()
    return render_template('edit.html', all_data=result)

@app.route('/update/<id>',  methods=['POST', 'GET'])
def update(id):
    if request.method == 'POST':
        nama = request.form['nama_edit']
        alamat = request.form['alamat_edit']
        no_hp = request.form['no_hp_edit']
        plat = request.form['plat_edit']
        cur = mysql.connection.cursor()
        cur.execute('UPDATE tb_identifikasi SET nama=%s, alamat=%s, no_hp=%s, plat=%s WHERE id=%s', (nama, alamat, no_hp, plat, id,))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('all_data'))

@app.route('/hitung')
def hitung():
    files = []
    path_char = r"static/img/char_try//"

    for file_name in os.listdir(path_char):
        # construct full file path
        file = path_char + file_name
        if os.path.isfile(file):
            files.append(file)

    files = sorted(files)
    print(files)
    i = 0
    chars = []
    for file in files:
        img = cv2.imread(file, 0)
        blur = cv2.GaussianBlur(img, (7, 7), 0)
        binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img_resize = resize(binary, (25, 25, 3))
        l = [img_resize.flatten()]
        probability = model.predict_proba(l)
        char = categories[model.predict(l)[0]]
        chars.append(char)
        print(categories[model.predict(l)[0]], end=' ')

    return f'yes'

def sort_contours(cnts, reverse=False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts

if __name__ == "__main__":
    app.run(debug=True)
