#!/usr/bin/env python3

import os
import sys
import cv2
import shutil

def convert_yolo_to_keras(yolo_bbox, img_width, img_height, new_size):
    c, x_norm, y_norm, w_norm, h_norm = yolo_bbox
    c = int(c)

    # Convert to absolute coords
    x_center = x_norm * img_width
    y_center = y_norm * img_height
    width = w_norm * img_width
    height = h_norm * img_height

    x_min = x_center - (width / 2)
    y_min = y_center - (height / 2)
    x_max = x_center + (width / 2)
    y_max = y_center + (height / 2)

    # Scale to new image size
    scale_x = new_size / img_width
    scale_y = new_size / img_height

    new_x_min = x_min * scale_x
    new_y_min = y_min * scale_y
    new_x_max = x_max * scale_x
    new_y_max = y_max * scale_y

    # Clamp values
    new_x_min = max(0, min(new_size, new_x_min))
    new_y_min = max(0, min(new_size, new_y_min))
    new_x_max = max(0, min(new_size, new_x_max))
    new_y_max = max(0, min(new_size, new_y_max))

    return f"{c} {int(new_x_min)} {int(new_y_min)} {int(new_x_max)} {int(new_y_max)}\n"

def main():
    if len(sys.argv) != 4:
        print("Usage: python resize_yolo_to_keras.py <input_dir> <output_dir> <output_size>")
        return

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    output_size = int(sys.argv[3])

    images_input_dir = os.path.join(input_dir, "img")
    anno_input_dir = os.path.join(input_dir, "anno")
    labels_file = os.path.join(input_dir, "labels.txt")

    images_output_dir = os.path.join(output_dir, "img")
    anno_yolo_output_dir = os.path.join(output_dir, "anno_yolo")
    anno_keras_output_dir = os.path.join(output_dir, "anno_keras")

    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(anno_yolo_output_dir, exist_ok=True)
    os.makedirs(anno_keras_output_dir, exist_ok=True)

    if os.path.isfile(labels_file):
        for dest in [output_dir, anno_yolo_output_dir, anno_keras_output_dir]:
            shutil.copy(labels_file, dest)

    image_files = []
    for f in os.listdir(images_input_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            image_files.append(f)

    for file_name in image_files:
        image_path = os.path.join(images_input_dir, file_name)
        base_name = os.path.splitext(file_name)[0]
        txt_path = os.path.join(anno_input_dir, base_name + ".txt")

        image = cv2.imread(image_path)
        if image is None:
            continue

        original_height, original_width = image.shape[:2]

        resized_image = cv2.resize(image, (output_size, output_size))

        new_yolo_lines = []
        new_keras_lines = []

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                # YOLO: class x_center y_center width height
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                c, x_norm, y_norm, w_norm, h_norm = map(float, parts)
                c = int(c)

                # Convert to scaled YOLO format
                x_center_abs = x_norm * original_width
                y_center_abs = y_norm * original_height
                w_abs = w_norm * original_width
                h_abs = h_norm * original_height

                scale_x = output_size / original_width
                scale_y = output_size / original_height

                new_x_center_norm = (x_center_abs * scale_x) / output_size
                new_y_center_norm = (y_center_abs * scale_y) / output_size
                new_w_norm = (w_abs * scale_x) / output_size
                new_h_norm = (h_abs * scale_y) / output_size

                yolo_line = f"{c} {new_x_center_norm:.6f} {new_y_center_norm:.6f} {new_w_norm:.6f} {new_h_norm:.6f}\n"
                new_yolo_lines.append(yolo_line)

                keras_line = convert_yolo_to_keras([c, x_norm, y_norm, w_norm, h_norm], original_width, original_height, output_size)
                new_keras_lines.append(keras_line)

        output_image_path = os.path.join(images_output_dir, file_name)
        cv2.imwrite(output_image_path, resized_image)

        output_yolo_txt_path = os.path.join(anno_yolo_output_dir, base_name + ".txt")
        with open(output_yolo_txt_path, 'w') as f:
            f.writelines(new_yolo_lines)

        output_keras_txt_path = os.path.join(anno_keras_output_dir, base_name + ".txt")
        with open(output_keras_txt_path, 'w') as f:
            f.writelines(new_keras_lines)

if __name__ == "__main__":
    main()

