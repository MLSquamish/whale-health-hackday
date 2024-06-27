from typing import List, Iterable

import cv2

def video_frame_generator(video_path:str, frames:Iterable|None = None):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        exit()

    if frames is None:
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = range(length)

    for frame_number in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame at the specified frame number
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            print(f"Error: Unable to read frame {frame_number}.")
            exit()

        yield frame_number, frame

    # Release the VideoCapture object
    cap.release()


if __name__ == "__main__":
    for frame_number, frame in video_frame_generator(r"./data/2023-08-25_Phantom4_18_Gully_0015-003.MOV", []):
        cv2.imshow("video", frame)
        cv2.waitKey(1)