from onnxsim import simplify
import onnx


def onnx_simplify(onnx_path: str, onnx_sim_path: str):
    """
    onnx simplifier method, same as `python -m onnxsim onnx_path onnx_sim_path
    :param onnx_path: path to onnx model
    :param onnx_sim_path: path for output onnx simplified model
    """
    model_sim, check = simplify(model=onnx_path, perform_optimization=False)
    if not check:
        raise RuntimeError("Simplified ONNX model could not be validated")
    onnx.save_model(model_sim, onnx_sim_path)
    return onnx_sim_path
