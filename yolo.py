import cv2
import torch
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  

# Open video file
video_path = 'campus.mp4'  # Change this to your video path
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_video_path = 'campus_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
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

# Loop through each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break 
     
    results = model(frame)

    # Parse results
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

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  

        label = f'{class_name} {probability:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)

    # Optional 
    cv2.imshow('Detection in Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything once done
cap.release()
out.release()
cv2.destroyAllWindows()
