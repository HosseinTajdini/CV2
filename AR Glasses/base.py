import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
glasses = cv2.imread(r'd:/All_Project/python/CV2/AR Glasses/glasses.png', -1)
print(glasses.shape)  

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print('Failed to read frame')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        new_glasses_width = int(w * 1.2)
        new_glasses_height = int(h / 4)
        resized_glasses = cv2.resize(glasses, (new_glasses_width, new_glasses_height))
        gh, gw, gc = resized_glasses.shape 

        y_offset = int(h / 5)
        x_offset = int((gw - w) / 2)


        if y + y_offset + gh > frame.shape[0] or x - x_offset < 0 or x - x_offset + gw > frame.shape[1]:
            continue


        for i in range(gh):  
            for j in range(gw):  
                alpha = resized_glasses[i, j][3]
                if alpha != 0:
                    frame[y + y_offset + i, x - x_offset + j] = resized_glasses[i, j][:3]

    cv2.imshow('AR Glasses', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()
