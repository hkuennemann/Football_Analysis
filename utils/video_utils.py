"""Has the utilities to read in and save the video"""
# pylint: disable=no-member

import cv2

def read_video(video_path):
    """Reads in a video file and returns the video in form of frames"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames, output_video_path):
    """Saves the video to the specified path"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename = output_video_path,
                          fourcc = fourcc,
                          fps = 20.0,
                          frameSize = (output_video_frames[0].shape[1],
                                       output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
