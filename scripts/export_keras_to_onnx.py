import os
import onnx
import tensorflow as tf
import tf2onnx

def validate_onnx_model(path):
    try:
        model = onnx.load(path)
        onnx.checker.check_model(model)
        print(f"[VALID] ONNX model is valid: {path}")
    except Exception as e:
        print(f"[ERROR] ONNX model validation failed: {path}")
        print(e)

def export_keras_to_onnx(model_path, onnx_path, opset=12, input_shape=(1,42, 42, 3)):
    keras_model = tf.keras.models.load_model(model_path)
    input_spec = (tf.TensorSpec(shape=input_shape, dtype=tf.float32, name="input"),)

    model_proto, _ = tf2onnx.convert.from_keras(
        keras_model,
        input_signature=input_spec,
        opset=opset,
        output_path=onnx_path
    )

    if os.path.exists(onnx_path):
        print(f"[Keras] Exported to: {onnx_path}")
        validate_onnx_model(onnx_path)
    else:
        print("[Keras] Export failed or ONNX file not found.")

if __name__ == "__main__":
    keras_model_path = "models/webcam_tile_e199.keras"
    keras_onnx_output = "models/webcam_tile_e199.onnx"
    export_keras_to_onnx(keras_model_path, keras_onnx_output)

