import os
import re
import cv2
import numpy as np
import onnxruntime as ort
import copy
import time
from matplotlib import pyplot as plt

corner_model_path = "models/webcam_corners_yv10.onnx"
corner_sess = ort.InferenceSession(corner_model_path)

tile_model_path = "models/webcam_tile_e199.onnx"
tile_sess = ort.InferenceSession(tile_model_path)

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
    warped = cv2.warpPerspective((input_image).astype(np.float32), M, (640, 640))
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

def draw_tile(img, tiles_on_board, tile_idx):
    import copy
    import cv2

    grid_size = 15
    tile_px = 640 // grid_size

    row = tile_idx // grid_size
    col = tile_idx % grid_size

    x = col * tile_px + 10
    y = row * tile_px + 30

    color = (0, 255, 0)
    thickness = 2 
    font_size = 0.6

    img_tiles = copy.deepcopy(img)

    cv2.putText(
        img_tiles,                
        f"{CLASS_NAMES[tiles_on_board[tile_idx]]}",              
        (x, y),                  
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,                       
        color,             
        thickness                        
    )
    return img_tiles

def draw_tiles_all_cells(img, tiles_on_board):
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
    
        frame = cv2.resize(frame, (640, 640))
        input_image = frame.astype("float32")
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image /= 255 # MUST BE 0-1 FOR FIRST NETWORK
        input_image_expanded = np.expand_dims(input_image, axis=0)
        input_image_expanded = np.transpose(input_image_expanded , (0, 3, 1, 2))  # HWC -> CHW

        output = corner_sess.run(None, {"images": input_image_expanded})[0][0]
        
        # print(f"{'Index':<5} {'Conf':<8} {'Class':<6} {'Box [x1 y1 x2 y2]':<30}")
        # print("-" * 55)
        # for i, det in enumerate(output):
        #     if i < 10:
        #         x1, y1, x2, y2, conf, cls = det
        #         print(f"{i:<5} {conf:<8.3f} {int(cls):<6} [{x1:6.1f} {y1:6.1f} {x2:6.1f} {y2:6.1f}]")
        
        boxes = output[:, :4][:4]
        confidences = output[:, 4][:4]
        boxes = boxes[confidences >= 0.3]
        print(confidences)
    
        corners = get_true_corners(boxes)
        input_image *= 255 # MUST BE 0-255 FOR TILE CLASSIFICATION
        filled_cells = []
        tiles_on_board = np.full((225,), -1, dtype=int)

        if len(corners) == 4:
            img_transformed_board = orient_board(input_image, corners)
            img_transformed_board = cv2.cvtColor(img_transformed_board , cv2.COLOR_RGB2BGR)

            for i in range(0, num_steps):
                y = int((640 / num_steps) * i)
                for j in range(num_steps):
                    x = int((640 / num_steps) * j)

                    x1 = x
                    y1 = y
                    x2 = x + cell_sz
                    y2 = y + cell_sz
                    cell_idx = i * num_steps + j

                    cells[cell_idx] = img_transformed_board[y1:y2, x1:x2]

                    r, g, b = (60, 60, 60)
                    target_color = np.array([b, g, r])
                    tol = np.array([60, 60, 60])
                    lower = np.clip(target_color - tol, 0, 255)
                    upper = np.clip(target_color + tol, 0, 255)
                    mask = cv2.inRange(cells[cell_idx], lower, upper)
                    coverage = np.count_nonzero(mask) / mask.size
                    
                    if coverage > 0.5:
                        filled_cells.append(cell_idx)

            cv2.imshow("cell", cells[0].astype(np.uint8))

            cells = cells.astype("float32")
            
            # img_grid = draw_grid(img_transformed_board)
            # img_coords = draw_coords(img_grid)
            img_tiles = draw_grid(img_transformed_board).astype(np.uint8)

            # if filled_cells > 7:
            #     continue

            for cell_idx in filled_cells:
                cell = cells[cell_idx]
                cell = np.expand_dims(cell, axis=0)

                pred = tile_sess.run(None, {'input': cell})[0][0]

                predicted_idx = np.argmax(pred)
                confidence = np.max(pred)
                if confidence >= 0.75:
                    tiles_on_board[cell_idx] = predicted_idx
                    # print(f"{CLASS_NAMES[predicted_idx], confidence}")

                img_tiles = draw_tile(img_tiles, tiles_on_board, cell_idx).astype(np.uint8)

            cv2.imshow("board", img_tiles)

        overlay = frame.copy()
        # print(corners)
        for coord in corners:
            radius = 10
            color = (0, 0, 1)
            cv2.circle(overlay, coord, radius, color, -1)
        alpha = 0.5
        img_cornerhighlight = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.imshow("frame", img_cornerhighlight)

        k = cv2.waitKey(1)
        if k == 27:
            break

    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

