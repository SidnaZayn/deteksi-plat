
@app.route('/process', methods=['POST','GET'])
def process():
    filessss = glob.glob(r"C:\Users\ACER\PycharmProjects\FlaskTry\static\img\char\*.jpg")
    for file in filessss:
        os.remove(file)
    # Import Haar Cascade XML file
    haar_cascade = cv2.CascadeClassifier('./static/cascade2.xml')

    # Read car image and convert color to RGB

    plate_img_norgbb = cv2.imread(r"./static/img/datatest 2/plat 7.jpg")

    #plate_img_norgbb = cv2.imread("./static/img/camera/img0000.jpg")
    width = int(plate_img_norgbb.shape[1] * 2)
    height = int(plate_img_norgbb.shape[0] * 2)
    dim = (width, height)
    resized_image = cv2.resize(plate_img_norgbb, dim, interpolation=cv2.INTER_AREA)

    plate_overlays = resized_image.copy()
    plate_overlays2 = resized_image.copy()# Create overlay to display red rectangle of detected car plate
    plate_rects = []
    plate_rects = haar_cascade.detectMultiScale(plate_overlays, scaleFactor=1.1, minNeighbors=5)

    yo = all(map(lambda x: x is None, plate_rects))

    if yo == True:
        print("Posisi plat tidak diketahui")
        return render_template('undetected.html')

    else:
        x, y, w, h = plate_rects[0]
        cv2.putText(plate_overlays, 'Plat Nomer', (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (0 , 0 , 255), 2)
        cv2.rectangle(plate_overlays, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite("./static/img/cropped/plate.jpg", plate_overlays)

        carplate_img = plate_overlays2[y : y + h , x : x + w ]
        width = int(carplate_img.shape[1] )
        height = int(carplate_img.shape[0] )
        dim = (width , height)
        resized_image_plate = cv2.resize(carplate_img, dim, interpolation=cv2.INTER_AREA)
        #blur = cv2.GaussianBlur(resized_image, (7, 7), 0)
        cv2.imwrite("./static/img/cropped/plate_cropped.jpg", resized_image_plate)

        return render_template('process.html')




@app.route('/char', methods=['POST','GET'])
def char():
    result = cv2.imread("./static/img/cropped/plate_cropped.jpg")
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
                curr_num = gray[y :y + h, x :x + w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                # _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

    if len(crop_characters) <= 0:
        print(len(crop_characters))
        print("tidak ada yang terbaca")
        return render_template('undetected.html')

    else:
        # print("Detect {} letters...".format(len(crop_characters)))
        # fig = plt.figure(figsize=(10, 6))
        # plt.axis(False)
        # plt.imshow(test_roi)
        fig = plt.figure(figsize=(14, 4))
        grid = gridspec.GridSpec(ncols=len(crop_characters), nrows=1, figure=fig)

        for i in range(len(crop_characters)):
            fig.add_subplot(grid[i])
            plt.axis(False)
            plt.imshow(crop_characters[i], cmap="gray")
            fig.savefig(r'C:\Users\ACER\PycharmProjects\FlaskTry\static\img\char\characters.png')

            path = r"C:\Users\ACER\PycharmProjects\FlaskTry\static\img\char"

            cv2.imwrite(os.path.join(path, str(i) + '.jpg'), crop_characters[i])

    return render_template('char.html')

@app.route('/read', methods=['POST','GET'])
def read():
    files = glob.glob(r"C:\Users\ACER\PycharmProjects\FlaskTry\static\img\char\*.jpg")

    files = sorted(files)

    i = 0
    chars = []

    for file in files:
        img = cv2.imread(file,0)
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

        elif chars[4] == 'g':
            chars[4] = '6'
        elif chars[4] == 'q':
            chars[4] = '0'
        elif chars[4] == 'b':
            chars[4] = '8'
        elif chars[4] == 'z':
            chars[4] = '2'
        elif chars[4] == 'e':
            chars[4] = '6'
        elif chars[4] == 's':
            chars[4] = '3'
        elif chars[4] == 't':
            chars[4] = '7'

        elif chars[-1] == '6':
            chars[-1] = 'g'
        elif chars[-1] == '0':
            chars[-1] = 'q'
        elif chars[-1] == '8':
            chars[-1] = 'b'

    comma_separated = ''.join(chars)

    return render_template("read.html", chars = comma_separated)
