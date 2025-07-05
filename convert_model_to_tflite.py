import tensorflow as tf
from pathlib import Path
import tf2onnx

def convert_keras_to_onnx(keras_path, onnx_path, network_w, network_h):
    keras_path = Path(keras_path)
    onnx_path = Path(onnx_path)

    model = tf.keras.models.load_model(keras_path)
    print(model.summary)

    # Define custom input shape for export
    spec = (tf.TensorSpec([None, network_h, network_w, 3], tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, inputs_as_nchw=['input'])

    onnx_path.write_bytes(onnx_model.SerializeToString())
    print(f"Converted {keras_path} → {onnx_path}")

convert_keras_to_onnx("models/webcam_corners_e60.keras", "models/webcam_corners_e60.onnx", 640, 640)
print("DONE")
convert_keras_to_onnx("models/webcam_tile_e199.keras", "models/webcam_tile_e199.onnx", 42, 42)
