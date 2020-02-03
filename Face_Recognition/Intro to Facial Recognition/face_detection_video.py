#import libraries
import cv2
import face_recognition

#read video and get lenght
input_video = cv2.VideoCapture("zendaya.mp4")
lenght = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

#load sample image of the person that will be identified
image = face_recognition.load_image_file("images/zendaya.jpg")
#get face encoding
face_encoding = face_recognition.face_encodings(image)[0]
known_faces = [
    face_encoding,
]

'''now we need to extract frames, find and identify faces
and lastly create a new video combining the original frame
with the location of the face of the person annotated'''

face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    #grab single frame
    ret, frame = input_video.read()
    frame_number += 1

    #quit when video ends
    if not ret:
        break

    #convert from BGR (what OpenCV uses) to RGB (what face_rec uses)
    rgb_frame = frame[:, :, ::-1]

    #find all the faces and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_name = []
    for face_encoding in face_encodings:
        #see if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)

        name = None
        if match[0]:
            name = "Zendaya"

        face_names.append(name)

        #label the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if not name:
                continue

        #draw a boc around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255),2)

        #draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0,0,255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255,255,255),1)

        #write the resulting image to the output video file
        print("Writing frame {} / {}".format(frame_number, lenght))
        output_video.write(frame)

codec = int(input_film.get(cv2.CAP_PROP_FOURCC))
fps = int(input_film.get(cv2.CAP_PROP_FPS))
frame_width = int(input_film.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_film.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_movie = cv2.VideoWriter("zendaya.mp4", codec, fps, (frame_width,frame_height))

output_film.release()
input_film.release()
cv2.destroyAllWindows()
