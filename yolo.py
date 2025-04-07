import cv2
import argparse
import torch
import numpy as np
import time

# model size
# yolov5n
# yolov5s
# yolov5m
# yolov5l
# yolov5x
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device) 

parser = argparse.ArgumentParser(description="YOLOv5 object detection with FPS overlay.")
parser.add_argument('--input', type=str, required=True, help='Path to input video file')
parser.add_argument('--output', type=str, required=True, help='Path to save output video')
args = parser.parse_args()


# Open video file
video_path = args.input  # Change this to your video path
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_video_path = args.output
fourcc = cv2.VideoWriter_fourcc(*'XVID')  
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

def get_class_color(class_id):
    colors = [
        (0, 0, 255),   # Red for class 0
        (0, 255, 0),   # Green for class 1
        (255, 0, 0),   # Blue for class 2
        (255, 255, 0), # Yellow for class 3
        (0, 255, 255), # Cyan for class 4
        (255, 0, 255), # Magenta for class 5
        (128, 128, 0), # Olive for class 6
        (128, 0, 128), # Purple for class 7
        (0, 128, 128), # Teal for class 8
        (255, 165, 0), # Orange for class 9
    ]
    return colors[class_id % len(colors)]


prev_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break 
     
    results = model(frame)

    predictions = results.xywh[0].cpu().numpy()

    class_names = model.names
    
    for prediction in predictions:
        x_center, y_center, width, height, conf, class_id = prediction

        class_name = class_names[int(class_id)]
        probability = conf

        img_height, img_width, _ = frame.shape
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2) 
        y2 = int(y_center + height / 2) 

        color = get_class_color(int(class_id))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)  

        label = f'{class_name} {probability:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    current_time = time.time()
    fps = 1 / (current_time - prev_time + 1e-5) if prev_time else 0
    prev_time = current_time
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    out.write(frame)

    # Optional 
    # cv2.imshow('Detection in Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything once done
cap.release()
out.release()
cv2.destroyAllWindows()
