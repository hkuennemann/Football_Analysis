import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5 # Minimum distance of camera movement that is considered

        self.lk_params = dict(
            winSize = (15, 15), # Window size for the search
            maxLevel = 2, # Maximum level for the search
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) # Criteria for stopping
        )

        first_frame_greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_greyscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners = 50, # Maximum number of corners to be utilized for good features
            qualityLevel = 0.3, # Quality level for good features
            minDistance = 3, # Minimum distance between two corners
            blockSize = 7, # Search size for good features
            mask = mask_features, # Mask for good features (where to get them from)
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjustment = (position[0]-camera_movement[0], position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjustment'] = position_adjustment

    def get_camera_movement(self, frames, read_from_stub = False, stub_path = None):
        # Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0,0]]*len(frames)

        # Convert the image into a greyscale image
        frame_iter = iter(frames)  # Convert list to iterator to avoid indexing
        old_grey = cv2.cvtColor(next(frame_iter), cv2.COLOR_BGR2GRAY) ##
        ##old_grey = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_grey, **self.features)

        ##for frame_num in range(1, len(frames)):
        for frame_num, frame in enumerate(frame_iter, start=1):  ##
            ## new_grey = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) ##
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_grey, 
                                                          new_grey, 
                                                          old_features, 
                                                          None, 
                                                          **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, 
                                                                               new_features_point)
            
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(new_grey, **self.features)
            
            ## old_grey = new_grey.copy()
            old_grey = new_grey ##
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

             # Create a transparent overlay on the copied frame
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)

            # Blend the overlay onto the frame copy (preserves transparency effect)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Extract movement values
            x_movement, y_movement = camera_movement_per_frame[frame_num]

            # Add text annotations
            cv2.putText(img = frame, 
                        text = f'Camera Movement X: {x_movement:.2f}', 
                        org = (10, 30), 
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 1, 
                        color = (0, 0, 0), 
                        thickness = 3)
            cv2.putText(img = frame, 
                        text = f'Camera Movement Y: {y_movement:.2f}', 
                        org = (10, 60), 
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 1, 
                        color = (0, 0, 0), 
                        thickness = 3)
            
            output_frames.append(frame)
        
        return output_frames
