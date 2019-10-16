import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    cv2.imshow('frame',hsv)
    mask = cv2.inRange(hsv,(0,10,60),(20,150,255))
    #mask_arr = np.asarray(mask)
    #pt1,pt2 = cv2.pointPolygonTest(mask_arr,)
    #cv2.rectangle(mask_arr,color = (0,255,0),thickness =3)
    res = cv2.bitwise_and(hsv,hsv,mask=mask)
    ret = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
    conc= cv2.hconcat([hsv, ret])
    r = cv2.selectROI(conc)
    #imCrop = hsv[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    cv2.imshow('side', conc)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

