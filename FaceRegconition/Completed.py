import cv2
#import sys
#cascPath = sys.argv[0]
faceCascade = cv2.CascadeClassifier("F:\\opencv\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)
#fps = video_capture.get(cv2.CAP_PROP_FPS)
#fr_w = int(video_capture.get(3));fr_h = int(video_capture.get(4))
#out =cv2.VideoWriter('d:/outpy.mp4',cv2.VideoWriter_fourcc('W','M','V','2'), fps, (fr_w,fr_h))
#out1 =cv2.VideoWriter('outpy1.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 30, (fr_w,fr_h))
#out2 =cv2.VideoWriter('outpy2.avi',cv2.VideoWriter_fourcc(*'MJPG'), 30, (fr_w,fr_h))
while True:
    ret,frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,1.1,5)
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    if ret == True:
        #out.write(frame); out1.write(frame); out2.write(frame);
        cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

