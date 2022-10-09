from asyncore import read
import numpy as np
import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

cascade = cv2.CascadeClassifier('//haarcascade_russian_plate_number.xml')

states = {"AN":"Andaman and Nicobar", "AP":"Andhra Pradesh"}

def extract_num(img_name):
    global read
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in nplate:
        a,b = (int(0.02*img.shape[0]), int(0.025*img.shape[1]))
        plate = img[y+a:y+h-a, x+b:x+w-b, :]
        kernel = np.ones((1,1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.imread('image1.jpg')
        (thresh, plate) = cv2.threshold(plate_gray, 127,255,cv2.ADAPTIVE_THRESH_MEAN_C)

        read = pytesseract.image_to_string(plate)
        read = ''.join(e for e in read if e.isalnum())
        stat = read[0:2]
        try:
            print("CAR BELONGS TO", states[stat])
        except:
            print("STATE NOT RECOGNISED")
        print(read)
        cv2.rectangle(img, (x,y), (x+w, y+h), (51,51,255), 2)
        cv2.rectangle(img, (x, y-40), (x+w, y), (51,51,255), -1)
        cv2.putText(img,read, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,5,2)
        cv2.imshow('Plate', plate)
    cv2.imshow("Result", img)
    cv2.imwrite('result,jpg',img)
    cv2.waitkey(0)
    cv2.destroyAllWindows()

extract_num('C://Users//Vishal Srivastava//Desktop//Number Plate Detection//image1.jpg')

