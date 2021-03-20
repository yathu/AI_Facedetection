import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('Robert_Downey_Jr.jpeg')


gray_scale_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);

face_cordinates = trained_face_data.detectMultiScale(gray_scale_img)

(x,y,w,h) = face_cordinates[0];

cv2.rectangle(img,(x,y),(w,h))

cv2.imshow('image showing...',gray_scale_img)

cv2.waitKey()

