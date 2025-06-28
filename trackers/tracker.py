# trackers/tracker.py

import os
import cv2
import numpy as np
import pickle
from ultralytics import YOLO
import supervision as sv
from utils import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detection_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect(frames)
        tracks = {"players": [], "referee": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper to player
            for i, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[i] = cls_names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv["referee"]:
                    tracks["referee"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_annotations(self, frames, tracks):
        annotated_frames = []
        for frame_num, frame in enumerate(frames):
            if frame is None or frame_num >= len(tracks["players"]):
                annotated_frames.append(None)
                continue

            for track_id, data in tracks["players"][frame_num].items():
                frame = self.draw_ellipse(frame, data["bbox"], color=(0, 255, 0), track_id=track_id)

            for track_id, data in tracks["referee"][frame_num].items():
                frame = self.draw_ellipse(frame, data["bbox"], color=(0, 0, 255), track_id=track_id)

            if 1 in tracks["ball"][frame_num]:
                frame = self.draw_triangle(frame, tracks["ball"][frame_num][1]["bbox"], color=(0, 255, 255))

            annotated_frames.append(frame)
        return annotated_frames

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-4.5,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_react = x_center - rectangle_width // 2
        x2_react = x_center + rectangle_width // 2
        y1_react = y2 - rectangle_height // 2 + 15
        y2_react = y1_react + rectangle_height

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_react), int(y1_react)),
                          (int(x2_react), int(y2_react)),
                          color,
                          cv2.FILLED)

            x1_text = x1_react + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_react + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

# Utility functions
def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y

def get_bbox_width(bbox):
    x1, _, x2, _ = bbox
    return int(x2 - x1)
