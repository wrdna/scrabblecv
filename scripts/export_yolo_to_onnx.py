import os
import onnx
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationDataReader
)

def validate_onnx_model(path):
    try:
        model = onnx.load(path)
        onnx.checker.check_model(model)
        print(f"[VALID] ONNX model is valid: {path}")
    except Exception as e:
        print(f"[ERROR] ONNX model validation failed: {path}")
        print(e)

class DummyDataReader(CalibrationDataReader):
    def __init__(self, input_name, shape, num_samples=5):
        self.input_name = input_name
        self.shape = shape
        self.data_iter = iter([
            {input_name: np.random.rand(*shape).astype(np.float32)}
            for _ in range(num_samples)
        ])
    
    def get_next(self):
        return next(self.data_iter, None)

def quantize_onnx_model(input_path, output_path, input_name, input_shape):
    print(f"[Quantizing] {input_path} → {output_path}")
    reader = DummyDataReader(input_name=input_name, shape=input_shape)

    quantize_static(
        model_input=input_path,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )

    print(f"[Quantized] Saved to: {output_path}")
    validate_onnx_model(output_path)

def export_yolo_to_onnx(model_path, opset=12, quantize=False):
    yolo_model = YOLO(model_path)
    result = yolo_model.export(format="onnx", opset=opset, imgsz=640, dynamic=False, simplify=False)
    onnx_path = result

    if not (onnx_path and os.path.exists(onnx_path)):
        print("[YOLO] Export failed or ONNX path not found.")
        return

    print(f"[YOLO] Exported to: {onnx_path}")
    validate_onnx_model(onnx_path)

    if quantize:
        quantized_path = os.path.splitext(onnx_path)[0] + "_int8_qdq.onnx"
        quantize_onnx_model(onnx_path, quantized_path, input_name="images", input_shape=(1, 3, 640, 640))

if __name__ == "__main__":
    yolo_model_path = "models/webcam_corners_yv10.pt"

    QUANTIZE = False
    export_yolo_to_onnx(yolo_model_path, quantize=QUANTIZE)

