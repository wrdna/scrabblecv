import os
import numpy as np
import tensorflow as tf

def convert_keras_to_tflite(model_path, tflite_path, quantize=False):
    print(f"Loading Keras model from: {model_path}")
    keras_model = tf.keras.models.load_model(model_path)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    
    if quantize:
        print("Applying post-training quantization")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"TFLite model saved to: {tflite_path}")
    print(f"Model size: {tflite_size:.2f} MB")

def validate_tflite_model(tflite_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"[VALID] TFLite model successfully loaded: {tflite_path}")
        print(f"Input details: {input_details}")
        print(f"Output details: {output_details}")
        return True
    except Exception as e:
        print(f"[ERROR] TFLite model validation failed: {tflite_path}")
        print(e)
        return False

def functional_check(keras_model, tflite_path, batch_sz=1, tol=1e-2):
    print(f"[FUNC‑CHK]  batch={batch_sz}, tol={tol}")
    x = np.random.rand(batch_sz, 42, 42, 3).astype("float32")

    ref = keras_model.predict(x, verbose=0)

    intr = tf.lite.Interpreter(model_path=tflite_path)
    intr.allocate_tensors()
    iidx = intr.get_input_details()[0]["index"]
    oidx = intr.get_output_details()[0]["index"]

    intr.set_tensor(iidx, x)
    intr.invoke()
    out = intr.get_tensor(oidx)

    abs_err = np.max(np.abs(ref - out))
    match = np.mean(ref.argmax(-1) == out.argmax(-1))

    print(f"   max |Δ| : {abs_err:.4e}")
    print(f"   argmax match : {match*100:.1f} %")

    if abs_err > tol:
        print("exceeds tolerance")

if __name__ == "__main__":
    keras_model_path = "models/webcam_tile_e199.keras"
    tflite_output_path = "models/webcam_tile_e199.tflite"
    tflite_quantized_path = "models/webcam_tile_e199_quantized.tflite"

    # export RAW fp32
    convert_keras_to_tflite(keras_model_path, tflite_output_path, quantize=False)
    if validate_tflite_model(tflite_output_path):
        model = tf.keras.models.load_model(keras_model_path)
        functional_check(model, tflite_output_path, tol=1e-5)

    # export fp16
    convert_keras_to_tflite(keras_model_path, tflite_quantized_path, quantize=True)
    if validate_tflite_model(tflite_quantized_path):
        model = tf.keras.models.load_model(keras_model_path)  # reload to avoid state changes
        functional_check(model, tflite_quantized_path, tol=1e-2)
