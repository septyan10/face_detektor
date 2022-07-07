import cv2 as cv

#start web cam
capture = cv.VideoCapture(0) # 0 for web-cam

#read the harr_face_detect_classifier.xml
harr_cascade = cv.CascadeClassifier("harr_face_detect_classifier.xml")

while True:
    #read video frame by frame
    isTrue, frame= capture.read()

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face_cords = harr_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=1)

    #draw rectange over faces
    for x, y, w, h in face_cords:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255,0), thickness=2)

    #show face detect Video
    cv.imshow("Detect face live Video", frame)

    #press e to exit
    if cv.waitKey(20) ==ord("q"):
        break

capture.release()
capture.destroyAllWindows()