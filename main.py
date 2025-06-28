from utils import read_video, save_video
from trackers.tracker import Tracker
import os

def main():
    video_paths = [
        "Input_videos/15sec_input_720p.mp4",
        "Input_videos/broadcast.mp4",
        "Input_videos/tacticam.mp4"
    ]

    tracker = Tracker("models/best.pt")

    for path in video_paths:
        video_name = os.path.splitext(os.path.basename(path))[0]
        print(f"[INFO] Processing: {video_name}")

        frames = read_video(path)
        if not frames:
            print(f"[WARNING] No frames in {path}. Skipping.")
            continue

        tracks = tracker.get_object_tracks(frames, read_from_stub=False)
        output_frames = tracker.draw_annotations(frames, tracks)

        output_path = f"output_videos/{video_name}_annotated.avi"
        save_video(output_frames, output_path)

if __name__ == "__main__":
    main()
