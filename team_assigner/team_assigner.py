import numpy as np
import cv2

from utils import get_center_of_bbox

    
class TeamAssigner():
    def __init__(self):
        self.best_line = None

    def draw_line(self, frames):
        for frame in frames:
            best_line = self.find_middle_line(frame)

    def get_line_metrics(self):
        x1, y1, x2, y2 = self.best_line[0]

        # Calculate slope (m) and intercept (c) of the middle line
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
        else:
            # In case of Perfectly vertical line
            slope = float('inf')
            intercept = 0  
        
        return slope, intercept
    
    def assign_color(self, bbox):
        x, _ = get_center_of_bbox(bbox)
        y = bbox[3]

        slope, intercept = self.get_line_metrics()

        # TODO: Assign colors based on player jersey
        team_a_color = (255, 0, 0) 
        team_b_color = (0, 0, 255) 

        if slope != float('inf'):  # Non-vertical case
            y_on_line = slope * x + intercept
        else:  # Vertical line case
            y_on_line = y  # Compare directly on x-axis

        # Assign colors
        if y < y_on_line:  
            player_color = team_a_color
        else: 
            player_color = team_b_color

        return player_color
    

    def find_middle_line(self, frames, frame_num):
        # Convert to grayscale
        frame = frames[frame_num]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Enhance edges using Gaussian blur + Canny
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # Hough Line Transform with added constraints
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=200, minLineLength=100, maxLineGap=20)
        
        # Draw filtered vertical lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Filter for near-vertical lines
                angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
                if (70 < angle < 80) and (900 < x1 < 1100) and (900 < x2 < 1100):   # Accept only near-vertical near-center lines
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    self.best_line = line
        else: # Repeat with next frame if line is not detected (Player blocking middle line)
            frame_num += 1
            self.find_middle_line(frames, frame_num)
    