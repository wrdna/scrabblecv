import cv2
import os
import re

OUTPUT_DIR = "data/corners/diamond_captures/img/"
output_img_stem = "dia_cap_"

def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def get_next_capture_index(output_dir, output_file_stem):
    existing_files = os.listdir(output_dir)
    capture_numbers = []

    pattern = re.compile(rf"{output_file_stem}(\d+)\.jpg")
    
    for file in existing_files:
        match = pattern.match(file)
        if match:
            capture_numbers.append(int(match.group(1)))

    return max(capture_numbers) + 1 if capture_numbers else 0

def main():
    create_dir(OUTPUT_DIR)

    # Doesnt change to 640, must be resized :/
    cap = cv2.VideoCapture(0, cv2.CAP_VFW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Resolution: {frame_width}x{frame_height}")

    capture_count = get_next_capture_index(OUTPUT_DIR, output_img_stem)

    while True:
        status, frame = cap.read()
        
        if not status:
            print("Failed to grab frame")
            break

        frame = cv2.resize(frame, (640, 640))
        cv2.imshow('Camera', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord(' '):
            filename = f"{output_img_stem}{capture_count}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(filepath, frame)
            print(f"Image saved: {filepath}")
            capture_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

