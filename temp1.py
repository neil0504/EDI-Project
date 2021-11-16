import cv2
import numpy as np

cap = cv2.VideoCapture("F:\\Videos and Pics\\VID_20211001_072900.mp4")

# algo = cv2.createBackgroundSubtractorKNN()
# algo = cv2.bgsegm.createBackgroundSubtractorGMG()
algo = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    img = frame[350:, ]
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame, (3,3), 5)
    img_substract = algo.apply(blur)
    # gray = cv2.cvtColor(img_substract, cv2.COLOR_BGR2GRAY)
    # ret,thresh1 = cv2.threshold(img_substract, 50, 255, cv2.THRESH_BINARY)
    # cv2.imshow("Dilated Data", img_substract)
    # cv2.imshow("Dilated Data", thresh1)
    # dilate = cv2.dilate(img_substract, np.ones((5, 5)))
    # kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # kernal = np.ones((5,5), np.uint8)
    # dilateData = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernal)
    # dilateData = cv2.morphologyEx(img_substract, cv2.MORPH_CLOSE, kernal)
    # countour, h = cv2.findContours(img_substract, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # for i, c in enumerate(countour):
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     validate_countour = (w>=80) and (h>= 80)
    #     if not validate_countour:
    #         continue

    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (50,50,450,290)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    cv2.imshow("Dilated Data", img)

    cv2.imshow("Original", frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
