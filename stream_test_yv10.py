import cv2
import numpy as np
import tensorflow as tf
import copy
from ultralytics import YOLO

model_path = "models/webcam_corners_yv10.pt"
model = YOLO(model_path)

def get_true_corners(boxes):
    corners = []
    for box in boxes:
        print(box)
        x1, y1, x2, y2 = box
        center_x = int(x1 + ((x2 - x1) / 2))
        center_y = int(y1 + ((y2 - y1) / 2))
        center_coords = (center_x, center_y)
        corners.append(center_coords)

    return corners

def orient_board(input_image, corners):
    sums = [x + y for (x, y) in corners]
    diffs = [x - y for (x, y) in corners]
    top_left  = corners[np.argmin(sums)]
    top_right = corners[np.argmin(diffs)]
    bot_right = corners[np.argmax(sums)]
    bot_left  = corners[np.argmax(diffs)]

    src_pts = np.array([top_left, bot_left, bot_right, top_right], dtype=np.float32)

    dst_pts = np.array([
        [0, 0],
        [639, 0],
        [639, 639],
        [0, 639]
    ], dtype=np.float32)

    # Create and apply perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective((input_image).astype(np.uint8), M, (640, 640))
    warped = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)

    return warped

def main():
    class_names = ["corner"]
    cap = cv2.VideoCapture(0, cv2.CAP_VFW)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    
    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break
    
        input_image = cv2.resize(frame, (640, 640))
    
        # Inference on frame
        result = model(input_image, conf=0.1, max_det=4)[0]
    
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_ids = result.boxes.cls
        
        corners = get_true_corners(boxes)

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        # Drawing highlighted corners
        overlay = input_image.copy()
        for coord in corners:
            radius = 10
            color = (0, 0, 1)
            cv2.circle(overlay, coord, radius, color, -1)
        alpha = 0.5
        img_cornerhighlight = cv2.addWeighted(overlay, alpha, input_image, 1 - alpha, 0)
    
        if len(corners) == 4:
            img_orientedboard = orient_board(input_image, corners)
   
            # Drawing lines on grid
            num_steps = 15
            grid_color = (255, 0, 0)
            thickness = 2
            img_gridboard = copy.deepcopy(img_orientedboard)
            # Vertical gridlines
            for i in range(1, num_steps):
                x = int((640 / num_steps) * i)
                cv2.line(img_gridboard, (x, 0), (x, 639), grid_color, thickness)
            # Horizontal gridlines
            for i in range(1, num_steps):
                y = int((640 / num_steps) * i)
                cv2.line(img_gridboard, (0, y), (639, y), grid_color, thickness)
    
            cv2.imshow("Warped Board",img_gridboard)
            
        img_cornerhighlight = cv2.cvtColor((img_cornerhighlight).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow("Corner Detection", img_cornerhighlight)
    
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
