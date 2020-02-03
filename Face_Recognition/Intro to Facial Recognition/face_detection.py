#import libraries
import cv2
import face_recognition

#get reference to webcam
video_capture = cv2.VideoCapture("/dev/video1")

#initialise variables
face_locations = []

while True:
    #grab single frame
    ret, frame = video_capture.read()
    if ret:
        #convert from BGR (which OpenCV uses) to RGB (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]
        #find all the faces in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        #display results
        for top, right, bottom, left in face_locations:
            #draw a box around the face
            cv2.rectangle(frame, (left,top), (right,bottom), (0,0,255), 2)

        #display resulting image
        cv2.imshow('Video', frame)
        #q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
