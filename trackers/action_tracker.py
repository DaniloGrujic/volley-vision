from ultralytics import YOLO
import pickle
import cv2
from utils import get_bbox_width, get_center_of_bbox


class ActionTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_thresholds = {
            0: 0.1, # attack
            1: 0.1, # block
            2: 0.6, # defence
            3: 0.1, # serve
            4: 0.1, # set
            5: 0.1  # ball
            } 

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        action_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                action_detections = pickle.load(f)
            return action_detections
        
        for frame in frames:
            action_dict = self.detect_frame(frame)
            action_detections.append(action_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(action_detections, f)

        return action_detections
    
    def detect_frame(self, frame):
        results = self.model.predict(frame)[0]
        id_name_dict = results.names

        action_list = []
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            conf = box.conf.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]

            if object_cls_name != 'ball' and conf > self.class_thresholds[object_cls_id]:
                action_list.append([object_cls_name, result])
            
        return action_list
    
    def draw_actions_info(self, video_frames, action_detections):
        output_video_frames = []
        for frame, action_list in zip(video_frames, action_detections):

            for action in action_list:
                action_name = action[0]
                bbox = action[1]
         
                self.draw_action_info(frame, bbox, (100, 100, 100), action_name)

        return output_video_frames


    def draw_action_info(self, frame, bbox, color, action_name=None):
        # Coorinates based on detected action bounding box
        x2 = int(bbox[1])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        width = max(0, width)

        rectangle_width = 100
        rectangle_hight = 20

        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (x2 - rectangle_hight // 2) + 15
        y2_rect = (x2 + rectangle_hight // 2) + 15

        if action_name is not None:
            # Draw rectangle
            cv2.rectangle(
                frame, 
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
                )

            x1_text = x1_rect + 11

            cv2.putText(
                frame, 
                f"{action_name}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
                )
        
        return frame