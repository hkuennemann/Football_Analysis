"""A module to track objects in a video"""
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    """A class to track objects in a video"""
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object, objects_tracks in tracks.items():
            for frame_num, track in enumerate(objects_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate(method='linear', limit_direction='both')

        # Backfilling for first few frames
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        """Detects objects in the frames"""
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_detections = self.model.track(batch_frames, conf = 0.1)
            detections += batch_detections
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """Returns the tracks of the objects in the video"""
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names # format: {0: player, 1: goalkeeper, etc.}
            cls_names_inv = {v:k for k, v in cls_names.items()} # format: {player: 0, goalkeeper: 1, etc.}

            # Convert to supervsion Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper to player
            for object_ind, class_id in enumerate(detection.boxes.cls.tolist()):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            # Track the objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox': bbox}
            
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipes(self, frame, bbox, color, track_id = None):
        """Draws an ellipse on the frame"""
        y2 = int(bbox[3])

        # Get bbox center and width
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(img = frame, 
                    center = (x_center, y2), 
                    axes = (int(width), int(.35*width)), 
                    angle = 0.0, 
                    startAngle = -45, # a bit of the circle will not be drawn
                    endAngle = 235, # a bit of the circle will not be drawn
                    color = color, 
                    thickness = 2,
                    lineType = cv2.LINE_4
                    )
        
        rectangle_width = 40
        rectangle_height = 20

        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(img = frame, 
                          pt1 = (int(x1_rect), int(y1_rect)), 
                          pt2 = (int(x2_rect), int(y2_rect)), 
                          color = color, 
                          thickness = cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(img = frame,
                        text = str(track_id), 
                        org = (int(x1_text), int(y1_rect + 15)), 
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 0.6, 
                        color = (0, 0, 0), 
                        thickness = 2)
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y], 
            [x - 10, y - 20], 
            [x + 10, y - 20]])
        
        cv2.drawContours(image = frame, 
                         contours = [triangle_points], 
                         contourIdx = 0, 
                         color = color, 
                         thickness=cv2.FILLED)
        cv2.drawContours(image = frame, 
                         contours = [triangle_points], 
                         contourIdx = 0, 
                         color = (0,0,0), 
                         thickness=2) # boarder

        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Retrieve team ball control till frame
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]

        # Get number of times each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        # Getting the stats
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        # Draw text
        cv2.putText(img = frame,
                    text = f'Team 1 Ball Control: {team_1 * 100:.2f}%',
                    org = (1400, 900),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1,
                    color = (0, 0, 0),
                    thickness = 3)
        cv2.putText(img = frame,
                    text = f'Team 2 Ball Control: {team_2 * 100:.2f}%',
                    org = (1400, 950),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1,
                    color = (0, 0, 0),
                    thickness = 3)
        
        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get('team_color', (0, 0, 255))
                frame = self.draw_ellipes(frame = frame, 
                                          bbox = player['bbox'], 
                                          color=color, 
                                          track_id = track_id)
                
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame = frame, 
                                               bbox = player['bbox'], 
                                               color=(0, 255, 255))
                
            # Draw referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipes(frame = frame, 
                                          bbox = referee['bbox'], 
                                          color=(0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame = frame, 
                                           bbox = ball['bbox'], 
                                           color=(0, 255, 0))
            
            # Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)
        
        return output_video_frames
