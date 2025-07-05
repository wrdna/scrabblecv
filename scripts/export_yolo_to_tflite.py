import os
from ultralytics import YOLO
from pathlib import Path

def validate_yolo_tflite(model_path):
    try:
        tflite_model = YOLO(model_path)
        results = tflite_model("https://ultralytics.com/images/bus.jpg", verbose=False)
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        
        print(f"[VALID] TFLite model validated: {model_path} (Size: {model_size:.2f} MB)")
        print(f"Test inference successful - detected {len(results[0].boxes)} objects")
        return True
    except Exception as e:
        print(f"[ERROR] TFLite model validation failed: {model_path}")
        print(e)
        return False

def export_yolo_to_tflite(model_path, half=False, int8=False):
    yolo_model = YOLO(model_path)
    model_name = Path(model_path).stem
    
    if int8:
        print(f"[YOLO] Exporting {model_name} to INT8 TFLite format...")
        result_path = yolo_model.export(format="tflite", int8=True, simplify=True)
    elif half:
        print(f"[YOLO] Exporting {model_name} to FP16 TFLite format...")
        result_path = yolo_model.export(format="tflite", half=True, simplify=True)
    else:
        print(f"[YOLO] Exporting {model_name} to FP32 TFLite format...")
        result_path = yolo_model.export(format="tflite", simplify=True)
    
    if not os.path.exists(result_path):
        print("[ERROR] Export failed or TFLite file not found.")
        return None
    
    print(f"[YOLO] Exported to: {result_path}")
    validate_yolo_tflite(result_path)
    return result_path

if __name__ == "__main__":
    yolo_model_path = "models/webcam_corners_yv10.pt"
    
    export_yolo_to_tflite(yolo_model_path)
    
    export_yolo_to_tflite(yolo_model_path, half=True)
    
    # Export to INT8 TFLite (smallest size, may have lower accuracy)
    # export_yolo_to_tflite(yolo_model_path, int8=True)

