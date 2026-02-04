import torch
from model import CancerNet
import os
import torch
import onnx
import onnxruntime
import onnxscript

VERSION = "v1"
model = CancerNet(30)
model.load_state_dict(torch.load(f"models/{VERSION}/model.pt"))
model.eval()

dummy = torch.randn(1, 30)

os.makedirs(f"models/{VERSION}", exist_ok=True)

torch.onnx.export(
    model,
    dummy,
    f"models/{VERSION}/model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}},
    opset_version=9

)

print("✅ ONNX model created")
