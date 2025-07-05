#!/usr/bin/env python3
import subprocess
import tensorflow as tf
from pathlib import Path

def onnx2tf_convert(onnx_path: str, saved_model_dir: str) -> None:
    cmd = [
        "onnx2tf",
        "-i", onnx_path,
        "--output_folder", saved_model_dir,
        "--output_signaturedefs",       
        "--replace_to_pseudo_operators",
        "--non_verbose",
        "--batch_size", "1",
    ]
    if subprocess.run(cmd).returncode:
        raise RuntimeError("onnx2tf conversion failed")
    print("Onnx2TF Successful")

def saved_model_to_tflite(saved_model_dir: str, tflite_path: str,
                          fp16: bool = True) -> None:
    print(f"[TF→TFLite] {saved_model_dir}  →  {tflite_path}")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if fp16:                                     
        converter.target_spec.supported_types = [tf.float16]

    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    Path(tflite_path).write_bytes(tflite_model)
    print("Success")

if __name__ == "__main__":
    onnx_path       = "models/webcam_corners_yv10.onnx"
    saved_model_dir = "models/webcam_corners_yv10_saved"
    tflite_path     = "models/webcam_corners_yv10_fp16.tflite"

    onnx2tf_convert(onnx_path, saved_model_dir)
    saved_model_to_tflite(saved_model_dir, tflite_path, fp16=True)
