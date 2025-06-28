![tacticam_annotated_output](https://github.com/user-attachments/assets/39b858ef-8825-4657-a6d3-5b6d1f7ed989)![broadcast_annotated_output](https://github.com/user-attachments/assets/23a601b1-53af-4839-8c89-66a30429552c)# ⚽ Multi-Video Player Detection and Tracking using YOLOv11 + Ultralytics

This project performs **player, referee, goalkeeper, and ball detection** and **tracking** 
on football match videos using the **YOLOv11 object detection model** combined with **Ultralytics**.
It supports **multiple input videos**, processes them individually, annotates frames, and 
saves the output as annotated videos.

## 📸 Screenshots pf output
![Frame 1](![15sec_input_720p_annotated_output](https://github.com/user-attachments/assets/7fdc39f9-670d-4f6a-a2e1-4e4fbb6e6be5))

![Frame 2](![broadcast_annotated_output](https://github.com/user-attachments/assets/8de7497c-06c7-4959-845a-c4c077353a5c))

![Frame 3](![tacticam_annotated_output](https://github.com/user-attachments/assets/88d6a9c4-af59-48f3-83cc-eeb30b2fa022))



## 📁 Project Structure

MACHINELEARNINGPROJ/
├── Input_videos/ # Folder containing input football videos
│ ├── 15sec_input_720p.mp4
│ ├── broadcast.mp4
│ └── tacticam.mp4
│
├── output_videos/ # Annotated video outputs will be saved here
│
├── models/
│ └── best.pt # Trained YOLOv11 model file
│
├── tracker_stubs/ # Optional: store pickled tracking data for reuse
│
├── utils/
│ └── video_utils.py # read_video() and save_video() functions
│
├── trackers/
│ ├── init.py
│ └── tracker.py # Tracker class using YOLO + Ultralytics
│
├── main.py # Main script to run tracking pipeline
├── README.md # Project documentation (this file)
└── requirements.txt # Python dependencies

## 🧠 Features

- 🧍 Detects multiple object classes: `player`, `referee`, `goalkeeper`, and `ball`
- 🔄 Converts goalkeeper → player for simplified tracking
- 🧠 Tracks objects using ByteTrack across frames
- 🖍️ Annotates players with ellipses and track IDs
- 🔺 Marks balls with triangles
- 🎥 Saves annotated video per input
