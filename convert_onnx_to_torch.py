# %%
# export torch model to folder
import onnx
import onnx2torch.node_converters
import torch
from onnx2torch import convert
import os

model_name = "RN50"

# Path to ONNX model
onnx_dir = "./onnx/" + model_name
onnx_path = os.path.join(onnx_dir, "image_model_1_reshaped.onnx")
# You can pass the path to the onnx model to convert it or...
onnx_model = onnx.load(onnx_path)
torch_model_1 = convert(onnx_model)
torch_model_1.eval()
input = torch.randn(50, 1, 2048)
# import pdb; pdb.set_trace()
print(torch_model_1(input))
torch_model_1.to_folder("./torch_model/")

# %% quantize
from utils import Utils
import qlinear
# print(torch_model_1)
model_int8 = torch.ao.quantization.quantize_dynamic(
    torch_model_1,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8) 
# print(model_int8)
Utils.replace_node( model_int8, 
                    # torch.nn.Linear, 
                    torch.ao.nn.quantized.dynamic.Linear, 
                    # qlinear.QLinearPerGrp, 
                    qlinear.QLinear, 
                    # (), {'device':'cpu', 'w_bit':4, 'group_size':32} )
                    (), {'device':'aie', 'kernel_x_shape': (1, 2048), 'kernel_y_shape': (2048, 2048)} )
for n, m in torch_model_1.named_modules():
            if isinstance(m, qlinear.QLinearPerGrp):
                print(f"Preparing weights of layer : {n}")
                m.device = "aie"
                m.quantize_weights()
# for n, m in torch_model_1.named_modules():
#     if isinstance(m, onnx2torch.node_converters.OnnxMatMul):
#         print(n)
# for node in torch_model_1.graph.nodes:
#     node.

# %% run model
print(torch_model_1(input))
