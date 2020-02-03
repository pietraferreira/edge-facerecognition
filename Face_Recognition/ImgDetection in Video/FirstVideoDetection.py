
from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

#create an instance of the VideoObjectDetection class
detector = VideoObjectDetection()
#set the model type to YOLOv3, which was the model downloaded
detector.setModelTypeAsYOLOv3()
#set the model path to the file path of the model file
detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
#load the model into the instance of the VideoObjectDetection class
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join( execution_path, "traffic-mini.mp4"),
                                output_file_path=os.path.join(execution_path, "traffic_mini_detected_1"),
                                frames_per_second=29, log_progress=True)
print(video_path)
