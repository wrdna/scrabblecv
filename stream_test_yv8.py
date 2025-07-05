import cv2
import numpy as np
import tensorflow as tf
import keras_cv
import keras
import copy
from keras_cv import bounding_box

model_path = "models/corners_e99.keras"
model = tf.keras.models.load_model(model_path)

def get_true_corners(boxes):
    corners = []
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
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
    # src_pts = np.array([top_left, top_right, bot_right, bot_left], dtype=np.float32)

    dst_pts = np.array([
        [0, 0],
        [639, 0],
        [639, 639],
        [0, 639]
    ], dtype=np.float32)

    # Create and apply perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective((input_image * 255).astype(np.uint8), M, (640, 640))
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
    
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(frame_rgb, (640, 640))
        input_image = input_image.astype("float32") / 255.0
        input_image_expanded = np.expand_dims(input_image, axis=0)
    
        # Inference on frame
        preds = model.predict(input_image_expanded)
    
        boxes = preds["boxes"][0]
        confidences = preds["confidence"][0]
        class_ids = preds["classes"][0]
   
        # NMS, get remaining boxes
        indices = tf.image.non_max_suppression(
            boxes,
            confidences,
            max_output_size=50,
            iou_threshold=0.2,
            score_threshold=0.2
        )
        selected_boxes = tf.gather(boxes, indices).numpy()
        selected_scores = tf.gather(confidences, indices).numpy()
        corners = get_true_corners(selected_boxes)

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
            
        img_cornerhighlight = cv2.cvtColor((img_cornerhighlight * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow("Corner Detection", img_cornerhighlight)
    
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
