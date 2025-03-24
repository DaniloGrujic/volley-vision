from ultralytics import YOLO
import pickle
import cv2

from utils import get_center_of_bbox, get_bbox_width

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)


    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        people_detections = {'player': [], 'referee': []}

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                people_detections = pickle.load(f)
                
            return people_detections

        for frame in frames:
            player_dict, referee_dict = self.detect_frame(frame)
            people_detections['player'].append(player_dict)
            people_detections['referee'].append(referee_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(people_detections, f)

        return people_detections
    

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names
        
        # Initialize player and referee dicts
        player_dict = {}
        referee_dict = {}

        # Loop over results (detected people)
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            conf = box.conf.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]

            # Filter players and referees
            if object_cls_name == 'player' and conf > 0.9:
                player_dict[track_id] = {'bbox': result}
            elif object_cls_name == 'refree' and conf > 0.5:
                referee_dict[track_id] = {'bbox': result}

        return player_dict, referee_dict
        
        
    def draw_ellipses(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            
            for track_id, player in player_dict.items():
                color = player.get('team_color', (0, 0, 0))
                self.draw_ellipse(frame, player['bbox'], color, track_id)

            output_video_frames.append(frame)

        return output_video_frames


    def draw_ellipse(self, frame, bbox, color, track_id=None):
        # Get coordinates for drawing ellipse 
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        width = max(0, width)

        # Draw ellipse
        cv2.ellipse(
            frame, 
            center=(x_center, y2), 
            axes=(int(0.8 * width), int(0.3 * width)), 
            angle=0.0, 
            startAngle=-45, 
            endAngle=200, 
            color=color, 
            thickness=3,
            lineType=cv2.LINE_4
            )
        
        # Metrics for ID rectangle
        rectangle_width = 40
        rectangle_hight = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_hight // 2) + 15
        y2_rect = (y2 + rectangle_hight // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame, 
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
                )

            x1_text = x1_rect + 11
            if int(track_id) > 99:
                x1_text -= 10

            cv2.putText(
                frame, 
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
                )
        
        return frame
    
    def anotate_referee(self, video_frames, referees_detections):
        output_video_frames = []
        for frame, referees_dict in zip(video_frames, referees_detections):
            
            for _, referee in referees_dict.items():
                # Coordinates for rectangle
                x_center, y_center = get_center_of_bbox(referee['bbox'])
                rectangle_width = 40
                rectangle_hight = 20
                x1_rect = x_center - rectangle_width // 2
                x2_rect = x_center + rectangle_width // 2
                y1_rect = y_center - 50 - rectangle_hight // 2
                y2_rect = y_center - 50 + rectangle_hight // 2
               
                cv2.rectangle(
                    frame, 
                    (int(x1_rect), int(y1_rect)),
                    (int(x2_rect), int(y2_rect)),
                    (255, 255, 0),
                    cv2.FILLED
                    )
                
                cv2.putText(
                frame, 
                "Ref",
                (int(x1_rect + 5), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
                )

            output_video_frames.append(frame)

        return output_video_frames
    