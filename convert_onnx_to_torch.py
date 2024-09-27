# %%
# export torch model to folder
import onnx
import onnx2torch.node_converters
import torch
from onnx2torch import convert
import os
import sys
import numpy as np

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

model_name = "RN50"

# Path to ONNX model
onnx_dir = "./output_models/" + model_name
onnx_path = os.path.join(onnx_dir, "text_model_reshaped.onnx" if len(sys.argv) < 2 else sys.argv[1])
output_path = os.path.join(onnx_dir, "text_model_dynamic_quant.pt" if len(sys.argv) < 3 else sys.argv[2])
# You can pass the path to the onnx model to convert it or...
onnx_model = onnx.load(onnx_path)
torch_model_1 = convert(onnx_model)
torch_model_1.eval()
import onnxruntime
session = onnxruntime.InferenceSession(
               onnx_path,
               providers=["CPUExecutionProvider"])
input_shape = session.get_inputs()[0].shape
input_name = session.get_inputs()[0].name
for i in range(0, len(input_shape)):
    if not isinstance(input_shape[i], int):
        input_shape[i] = 1
# input = torch.randint(1, 10, input_shape).int()
# input_data = to_numpy(input)
# onnx_result = session.run([], {input_name: input_data})
# print(onnx_result)
# print(torch_model_1)
# import pdb; pdb.set_trace()
# print(torch_model_1(input))
# torch_model_1.to_folder("./text_torch_model/")

# %% quantize
try:
    from utils import Utils
    import qlinear
except Exception:
    import setup
    from utils import Utils
    import qlinear
# print(torch_model_1)
model_int8 = torch.ao.quantization.quantize_dynamic(
    torch_model_1,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)
torch.save(model_int8, output_path)
# print(model_int8)
Utils.replace_node( model_int8, 
                    # torch.nn.Linear, 
                    torch.ao.nn.quantized.dynamic.Linear, 
                    # qlinear.QLinearPerGrp, 
                    qlinear.QLinear, 
                    # (), {'device':'cpu', 'w_bit':4, 'group_size':32} )
                    (), {'device':'aie', 'kernel_x_shape': (8, 2048), 'kernel_y_shape': (2048, 2048)} )
for n, m in torch_model_1.named_modules():
            if isinstance(m, qlinear.QLinearPerGrp):
                print(f"Preparing weights of layer : {n}")
                m.device = "aie"
                m.quantize_weights()
model_int8.eval()
# for n, m in torch_model_1.named_modules():
#     if isinstance(m, onnx2torch.node_converters.OnnxMatMul):
#         print(n)
# for node in torch_model_1.graph.nodes:
#     node.
print("Dynamic quantized model is saved to ", output_path)
# %% run model
# print(torch_model_1(input))
