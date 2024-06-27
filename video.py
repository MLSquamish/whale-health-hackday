from typing import List, Iterable

import cv2
import os

# Drop the video listed below in the data folder
VIDEO_PATH = r"./data/2023-08-25_Phantom4_18_Gully_0015-003.MOV"

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

    # Create the frames folder if it doesn't exist
    frames_folder = "data/frames"
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)
    
    # List of frames to extract
    frames_to_extract = list(range(1000, 1200, 50))

    for frame_number, frame in video_frame_generator(VIDEO_PATH, frames_to_extract):

        # Save the frame in the data/frames folder (to add the folder)
        frame_path = os.path.join("data/frames", f"0015_003_frame_{frame_number}.jpg")
        cv2.imwrite(frame_path, frame)
        
        cv2.waitKey(1)