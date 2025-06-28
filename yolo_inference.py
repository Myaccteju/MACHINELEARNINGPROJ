#Import All the required libraries
from ultralytics import YOLO

#Load the YOLO Model
model = YOLO("models/best.pt") #yolol=yolo large

#Object Detection
video_list = ["input_videos/tacticam.mp4",
              "input_videos/15sec_input_720p.mp4",
              "input_videos/broadcast.mp4"
]

for video in video_list:
    output = model.predict(source=video, save = True)

#Tracking
#for video in video_list:
  #  output = model.track(source=video, save = True, persist=True)