from ultralytics import YOLO
import pickle
import cv2
from collections import deque
import pandas as pd
import numpy as np
from utils import get_center_of_bbox, get_bbox_width

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.ball_trail = deque(maxlen=5)


    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]

        # List to pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections


    def detect_frame(self, frame):
        results = self.model.track(frame, conf=0.3, iou=0.3)[0]

        ball_dick = {}
        for box in results.boxes:
            
            if box.id:
                track_id = int(box.id.tolist()[0])
                result = box.xyxy.tolist()[0]

                ball_dick[track_id] = result

        return ball_dick
        
    def draw_ball_path(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            
            for track_id, bbox in ball_dict.items():
                self.draw_ball_and_tail(frame, bbox, (255, 255, 0), track_id)

            output_video_frames.append(frame)

        return output_video_frames
    

    def draw_ball_and_tail(self, frame, bbox, color, track_id=None):
        # Ball coorinates
        x_center, y_center = get_center_of_bbox(bbox)
        ball_center = (x_center, y_center)
        width = get_bbox_width(bbox)

        # Draw circle around ball
        cv2.circle(
            frame, 
            ball_center, 
            int(width // 2), 
            color, 
            thickness=2)

        self.ball_trail.append(ball_center)

        # Draw trajectory lines
        for i in range(1, len(self.ball_trail)):
            if self.ball_trail[i - 1] is None or self.ball_trail[i] is None:
                continue
            
            thickness = 2
            cv2.line(frame, self.ball_trail[i - 1], self.ball_trail[i], color, thickness)

        return frame