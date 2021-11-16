import cv2
import numpy as np
cap = cv2.VideoCapture("F:\\Videos and Pics\\VID_20211001_072900.mp4")
_, first_frame = cap.read()
# background = cv2.imread("F:\\Videos and Pics\\IMG_20211001_063133.jpg")
# background = background[350:, ]
# first_frame = first_frame[350:, ]
background = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
background = cv2.GaussianBlur(background, (23, 23), 0)

while True:
    ret, frame = cap.read()
    # frame = frame[350:, ]
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = cv2.absdiff(background, frame1)
    f = f[350:, ]
    _, frame2 = cv2.threshold(f, 100, 255, cv2.THRESH_BINARY)
    dilate = cv2.dilate(frame2, np.ones((5, 5)))
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilateData = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernal)
    dilateData = cv2.morphologyEx(dilateData, cv2.MORPH_CLOSE, kernal)
    countour, h = cv2.findContours(dilateData, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(countour):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_countour = (w>=80) and (h>= 80)
        if not validate_countour:
            continue

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Difference", frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
