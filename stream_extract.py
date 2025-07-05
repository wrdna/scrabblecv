import os
import re
import cv2
import numpy as np
import tensorflow as tf
import keras_cv
import keras
import copy
from keras_cv import bounding_box
from matplotlib import pyplot as plt

corner_model_path = "models/webcam_corners_e60.keras"
corner_model = tf.keras.models.load_model(corner_model_path)

tile_model_path = "models/webcam_tile_e199.keras"
tile_model = tf.keras.models.load_model(tile_model_path)

CLASS_NAMES = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","*"]

def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def get_next_capture_index(output_dir, output_file_stem):
    existing_files = os.listdir(output_dir)
    capture_numbers = []

    # Matching "0_letter_A.jpg"
    pattern = re.compile(rf"^(\d+)_?{re.escape(output_file_stem)}_.*\.jpg$")
    
    for file in existing_files:
        match = pattern.match(file)
        if match:
            capture_numbers.append(int(match.group(1)))

    return max(capture_numbers) + 1 if capture_numbers else 0

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

    # Using this order else transform is mirrored/flipped. 
    src_pts = np.array([top_left, bot_left, bot_right, top_right], dtype=np.float32)
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

def draw_grid(img):
    grid_color = (255, 0, 0)
    thickness = 1
    num_steps = 15
    img_grid = copy.deepcopy(img)

    # Vertical gridlines
    for i in range(1, num_steps):
        x = int((640 / num_steps) * i)
        cv2.line(img_grid, (x, 0), (x, 639), grid_color, thickness)

    # Horizontal gridlines
    for i in range(1, num_steps):
        y = int((640 / num_steps) * i)
        cv2.line(img_grid, (0, y), (639, y), grid_color, thickness)
    return img_grid 

def draw_coords(img):
    color = (0, 0, 255)
    thickness = 1 
    num_steps = 15
    img_coords = copy.deepcopy(img)
    for i in range(num_steps):
        y = int((640 / num_steps) * i) + 12 
        for j in range(num_steps):
            x = int((640 / num_steps) * j) + 4
            cv2.putText(
                img_coords,                
                f"{i},{j}",              
                (x, y),                  
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,                       
                color,             
                thickness                        
            )
    return img_coords

def draw_tiles(img, tiles_on_board):
    color = (0, 255, 0)
    thickness = 2 
    font_size = 0.6
    num_steps = 15
    img_tiles = copy.deepcopy(img)
    for i in range(num_steps):
        y = int((640 / num_steps) * i) + 10 
        for j in range(num_steps):
            x = int((640 / num_steps) * j) + 0 
            if tiles_on_board[i * num_steps + j] != -1:
                cv2.putText(
                    img_tiles,                
                    f"{CLASS_NAMES[tiles_on_board[i * num_steps + j]]}",              
                    (x, y),                  
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,                       
                    color,             
                    thickness                        
                )
    return img_tiles

def main():

    SAVE_CELL = True

    board_sz = 640
    num_steps = 15
    cell_sz = board_sz // 15
    cells = np.zeros((num_steps * num_steps, cell_sz, cell_sz, 3), dtype=np.uint8)

    tiles_on_board = np.zeros((num_steps * num_steps), dtype=np.uint8) 

    cap = cv2.VideoCapture(0, cv2.CAP_VFW)
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
        corner_preds = corner_model.predict(input_image_expanded, verbose=False)
        boxes = corner_preds["boxes"][0]
        confidences = corner_preds["confidence"][0]
        class_ids = corner_preds["classes"][0]
   
        # NMS
        indices = tf.image.non_max_suppression(
            boxes,
            confidences,
            max_output_size=20,
            iou_threshold=0.2,
            score_threshold=0.2
        )
        selected_boxes = tf.gather(boxes, indices).numpy()
        selected_scores = tf.gather(confidences, indices).numpy()
        corners = get_true_corners(selected_boxes)

        # When all corners detected, extract into array
        if len(corners) == 4:
            img_transformed_board = orient_board(input_image, corners)

            for i in range(0, num_steps):
                y = int((640 / num_steps) * i)
                for j in range(num_steps):
                    x = int((640 / num_steps) * j)

                    x1 = x
                    y1 = y
                    x2 = x + cell_sz
                    y2 = y + cell_sz

                    cells[i * num_steps + j] = img_transformed_board[y1:y2, x1:x2]

            confidence_threshold = 0.8
            tile_pred = tile_model.predict(cells, batch_size=225, verbose=False)
            
            predicted_indices = np.argmax(tile_pred, axis=1)
            confidence_scores = np.max(tile_pred, axis=1)
            high_conf_mask = confidence_scores >= confidence_threshold
            tiles_on_board = np.where(high_conf_mask, predicted_indices, -1)

            # img_grid = draw_grid(img_transformed_board)
            # img_coords = draw_coords(img_grid)
            img_tiles = draw_tiles(img_transformed_board, tiles_on_board)
            cv2.imshow("board", img_tiles)

        k = cv2.waitKey(1)
        if k == 27:
            break

    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

