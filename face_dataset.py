import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# enter numeric face id
face_id = input('\n Enter Student ID:- ')
face_name=input('\n Enter Student Name:- ')

print("\n [INFO] Initializing face capture....")
# Initialize individual face count
count = 0

while(True):

    ret, img = cam.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 10, 6)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2) 
        count += 1

        # Save image into the datasets folder
        cv2.imwrite("dataset/" + str(face_id) +'.'+ str(face_name)+ '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(1) & 0xff 
    if k == 1:
        break
    elif count >= 400: 
         break

# cleanup
print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()