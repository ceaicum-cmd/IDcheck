"""
tflite_utils.py
Utility module for running TensorFlow Lite models directly.

Useful for:
- Custom TFLite models
- Edge / optimized inference
- When you need more control than MediaPipe Tasks API

Requires: tflite-runtime
"""

from typing import Optional, List, Dict, Any
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False


class TFLiteModel:
    """
    Wrapper class for easy TFLite model inference.
    """

    def __init__(self, model_path: str, use_gpu: bool = False):
        if not TFLITE_AVAILABLE:
            raise ImportError(
                "tflite-runtime is not installed. "
                "Install it with: pip install tflite-runtime"
            )

        self.model_path = model_path
        self.interpreter = Interpreter(model_path=model_path)

        # Optional GPU delegate (works on some platforms)
        if use_gpu:
            try:
                gpu_delegate = load_delegate('libtensorflowlite_gpu_delegate.so')
                self.interpreter = Interpreter(
                    model_path=model_path,
                    experimental_delegates=[gpu_delegate]
                )
            except Exception:
                print("GPU delegate not available, falling back to CPU.")

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def get_input_shape(self) -> tuple:
        return tuple(self.input_details[0]['shape'])

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on the model.
        """
        input_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_index, input_data.astype(np.float32))
        self.interpreter.invoke()

        # Return first output by default
        output_index = self.output_details[0]['index']
        return self.interpreter.get_tensor(output_index)

    def predict_multi_output(self, input_data: np.ndarray) -> List[np.ndarray]:
        """
        Run inference and return all outputs.
        """
        input_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_index, input_data.astype(np.float32))
        self.interpreter.invoke()

        outputs = []
        for detail in self.output_details:
            outputs.append(self.interpreter.get_tensor(detail['index']))
        return outputs


def load_tflite_model(model_path: str, use_gpu: bool = False) -> TFLiteModel:
    """Convenience function to load a TFLite model."""
    return TFLiteModel(model_path, use_gpu=use_gpu)