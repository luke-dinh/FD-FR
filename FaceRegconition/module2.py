import numpy as np
import cv2
cap = cv2.VideoCapture('example.mp4')
if (cap.isOpened() == False): print("Unable to read camera ")
fps = cap.get(cv2.CAP_PROP_FPS)
fr_w = int(cap.get(3));fr_h = int(cap.get(4))
out =cv2.VideoWriter('d:/outpy.wmv',cv2.VideoWriter_fourcc('W','M','V','2'), fps, (fr_w,fr_h))
out1 =cv2.VideoWriter('outpy1.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30, (fr_w,fr_h))
out2 =cv2.VideoWriter('outpy2.avi',cv2.VideoWriter_fourcc(*'MJPG'), 30, (fr_w,fr_h))
lower = np.array([10, 0, 0], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")
while(True):
    ret, frame = cap.read()
    width =  int(frame.shape[1]*scale_percent / 100, 10)
    height = int(frame.shape[0] * scale_percent / 100, 10 )
    dim = (width, height)

    frame = cv2.resize(frame,dim, interpolation = cv2.INTER_NEAREST)
	# and determine the HSV pixel intensities that fall into
	# the speicifed upper and lower boundaries
    #frame = imutils.resize(frame, width = 400)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    
	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

	# blur the mask to help remove noise, then apply the
	# mask to the frame
    #skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask) #Hinh co mat

    
    contours ,_= cv2.findContours(skinMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        #if cv2.contourArea(contour) <= 50 :
            #continue
        x,y,w,h = cv2.boundingRect(contour)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255,0), 2)
        sub_face = frame[y:y+h, x:x+w]
        sub_face = cv2.GaussianBlur(sub_face,(35, 35),30)
        frame[y:y+h, x:x+w]= sub_face
        center = (x,y)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask",skinMask)
    if ret == True:
        out.write(frame); out1.write(frame); out2.write(frame);
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            break
        cap.release();out.release(); out1.release(); out2.release();
    key = cv2.waitKey(1)
    if key == 27:
        break
        
cap.release()
cv2.destroyAllWindows()
