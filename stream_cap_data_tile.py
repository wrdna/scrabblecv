import os
import re
import cv2
import numpy as np
import tensorflow as tf
import keras_cv
import copy
from keras_cv import bounding_box
from matplotlib import pyplot as plt

model_path = "models/webcam_corners_e60.keras"
model = tf.keras.models.load_model(model_path)

OUTPUT_DIR = "data/letters/normalized/"
OUTPUT_IMG_STEM = "letter"

# Global variables for access in the mouse callback
global_cells = None
global_cell_sz = 640 // 15
global_clicked_cell =  (None, None)

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

def draw_prev_clicked_cells(img, prev_clicked_cells):
    color = (0, 255, 0)
    thickness = 3 
    num_steps = 15
    img_prev_clicked = copy.deepcopy(img)
    for i in range(num_steps):
        y = int((640 / num_steps) * i) + 30 
        for j in range(num_steps):
            x = int((640 / num_steps) * j) + 10 
            if prev_clicked_cells[i, j] !=  0:
                cv2.putText(
                    img_prev_clicked,                
                    f"{prev_clicked_cells[i,j]}",              
                    (x, y),                  
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,                       
                    color,             
                    thickness                        
                )
    return img_prev_clicked

def click_cell(event, x, y, flags, param):
    global global_cells, global_cell_sz, global_clicked_cell
    if event == cv2.EVENT_LBUTTONDOWN and global_cells is not None:
        row = min(int(y / 640 * 15), 15 - 1)
        col = min(int(x / 640 * 15), 15 - 1)
        # if row < global_cells.shape[0] and col < global_cells.shape[1]:
        global_clicked_cell = (row, col)
        print(f"Clicked cell at row {row}, col {col} - x {x}, y {y}")

def main():
    global global_cells, global_cell_sz, global_clicked_cell

    SAVE_CELL = True

    board_sz = 640
    num_steps = 15
    cell_sz = board_sz // 15
    cells = np.zeros((num_steps, num_steps, cell_sz, cell_sz, 3), dtype=np.uint8)

    global_cell_sz = cell_sz
    prev_clicked_cells = np.zeros((num_steps, num_steps), dtype=object)

    cap = cv2.VideoCapture(0, cv2.CAP_VFW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    cv2.namedWindow("board")
    cv2.setMouseCallback("board", click_cell)

    create_dir(OUTPUT_DIR)
    capture_count = get_next_capture_index(OUTPUT_DIR, OUTPUT_IMG_STEM)

    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break
    
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(frame_rgb, (640, 640))
        input_image = input_image.astype("float32") / 255.0
        input_image_expanded = np.expand_dims(input_image, axis=0)
    
        # Inference on frame
        preds = model.predict(input_image_expanded, verbose=False)
        boxes = preds["boxes"][0]
        confidences = preds["confidence"][0]
        class_ids = preds["classes"][0]
   
        # NMS
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

        # When all corners detected, extract into array
        if len(corners) == 4:
            img_transformed_board = orient_board(input_image, corners)
            img_grid = draw_grid(img_transformed_board)
            img_coords = draw_coords(img_grid)
            img_prev_clicked = draw_prev_clicked_cells(img_coords, prev_clicked_cells)

            cv2.imshow("board", img_prev_clicked)

            for i in range(0, num_steps):
                y = int((640 / num_steps) * i)
                for j in range(num_steps):
                    x = int((640 / num_steps) * j)

                    x1 = x
                    y1 = y
                    x2 = x + cell_sz
                    y2 = y + cell_sz

                    cells[i, j] = img_transformed_board[y1:y2, x1:x2]

            # Must update global_cells for mouse callback
            global_cells = cells.copy()

        if global_cells is not None and \
           global_clicked_cell[0] is not None and \
           global_clicked_cell[1] is not None:
            cv2.imshow("clicked_cell", global_cells[global_clicked_cell])

            if SAVE_CELL:
                label = None
                while(True): 
                    k = cv2.waitKey(33)

                    # -1 Default, 225 L-Shift
                    if k == -1 or \
                       k == 225:
                        continue
                    elif k == 27:
                        pass
                    else: 
                        label = chr(k).upper()
                        
                        if not label:
                            continue
                        
                        prev_clicked_cells[global_clicked_cell] = label

                        if label == '*':
                            label = "BLANK"

                        filename = f"{capture_count}_{OUTPUT_IMG_STEM}_{label}.jpg"
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        cv2.imwrite(filepath, global_cells[global_clicked_cell])
                        capture_count += 1
                        print(f"CELL {global_clicked_cell} LABELED {label} -- SAVED TO {filepath}")
                    
                    global_clicked_cell = (None, None)
                    global_cells = None

                    cv2.destroyWindow("clicked_cell")

                    break

        k = cv2.waitKey(1)
        if k == 27:
            break
        if k == ord("r"):
            prev_clicked_cells = np.zeros((num_steps, num_steps), dtype=object)

    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

