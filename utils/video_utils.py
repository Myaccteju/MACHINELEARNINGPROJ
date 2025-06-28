#Import All the Required Libraries
import cv2

#Create a Read Video Function
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

#Create a Save Video Function
def save_video(output_video_frames, output_video_path):
    # Filter out None frames before accessing .shape
    output_video_frames = [frame for frame in output_video_frames if frame is not None]

    if not output_video_frames:
        print("[ERROR] No valid frames to save.")
        return

    # Get dimensions from the first valid frame
    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))

    for frame in output_video_frames:
        out.write(frame)

    out.release()
    print(f"[INFO] Video saved to {output_video_path}")
