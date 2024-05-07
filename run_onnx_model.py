# %%
# load original model for preprocess
from PIL import Image
import open_clip
import gc
# model_name = "ViT-bigG-14"
# pretrain_dataset = "laion2b_s39b_b160k"
model_name = "ViT-B-32"
pretrain_dataset = "laion2b_s34b_b79k"
model_path = 'D:\\Models\\open-clip\\CLIP-ViT-B-32-laion2B-s34B-b79K\\open_clip_pytorch_model.bin'
# model_name = "convnext_large_d_320"
# pretrain_dataset = "laion2b_s29b_b131k_ft_soup"
# model_path = 'D:\\Models\\open-clip\\CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup\\open_clip_pytorch_model.bin'
_, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_path, cache_dir="./model_cache")
tokenizer = open_clip.get_tokenizer(model_name)

image = preprocess(Image.open("./demo.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])
print("text tokens: ", text)
gc.collect()
import numpy as np

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# %%
# run quantized model on NPU
import onnxruntime
import os
# Add user imports
# ...

# Load inputs and perform preprocessing
# ...

# Create an inference session using the Vitis AI execution provider

onnx_dir = "./onnx/"+model_name+"_"+pretrain_dataset
onnx_path = os.path.join(onnx_dir, "image_model_quantized.onnx")
config_path = 'C:\\Users\\austi\\Downloads\\ryzen-ai-sw-1.1\\ryzen-ai-sw-1.1\\voe-4.0-win_amd64\\vaip_config.json'
session = onnxruntime.InferenceSession(
               onnx_path,
               providers=["VitisAIExecutionProvider"],
               provider_options=[{"config_file":config_path}])

input_shape = session.get_inputs()[0].shape
input_name = session.get_inputs()[0].name

# Load inputs and do preprocessing by input_shape
input_data = to_numpy(image)
npu_qresult = session.run([], {input_name: input_data})
print("Quantized: ", npu_qresult)

# %%
# run non-quantized model on NPU
onnx_path = os.path.join(onnx_dir, "image_model.onnx")
config_path = 'C:\\Users\\austi\\Downloads\\ryzen-ai-sw-1.1\\ryzen-ai-sw-1.1\\voe-4.0-win_amd64\\vaip_config.json'
session = onnxruntime.InferenceSession(
               onnx_path,
               providers=["VitisAIExecutionProvider"],
               provider_options=[{"config_file":config_path}])

input_shape = session.get_inputs()[0].shape
input_name = session.get_inputs()[0].name

# Load inputs and do preprocessing by input_shape
input_data = to_numpy(image)
npu_result = session.run([], {input_name: input_data})
print('Non-quantized: ', npu_result)

# %%
# run non-quantized model on CPU
onnx_path = os.path.join(onnx_dir, "image_model.onnx")
config_path = 'C:\\Users\\austi\\Downloads\\ryzen-ai-sw-1.1\\ryzen-ai-sw-1.1\\voe-4.0-win_amd64\\vaip_config.json'
session = onnxruntime.InferenceSession(
               onnx_path,
               providers=["CPUExecutionProvider"])

input_shape = session.get_inputs()[0].shape
input_name = session.get_inputs()[0].name

# Load inputs and do preprocessing by input_shape
input_data = to_numpy(image)
result = session.run([], {input_name: input_data})
print('Non-quantized On CPU: ', result)
# %%
# run non-quantized model on NPU
onnx_path = os.path.join(onnx_dir, "text_model.onnx")
config_path = 'C:\\Users\\austi\\Downloads\\ryzen-ai-sw-1.1\\ryzen-ai-sw-1.1\\voe-4.0-win_amd64\\vaip_config.json'
session = onnxruntime.InferenceSession(
               onnx_path,
               providers=["VitisAIExecutionProvider"],
               provider_options=[{"config_file":config_path}])

input_shape = session.get_inputs()[0].shape
input_name = session.get_inputs()[0].name

# Load inputs and do preprocessing by input_shape
input_data = to_numpy(text[0].reshape([1, -1]))
cpu_result = session.run([], {input_name: input_data})
print('CPU Non-quantized: ', cpu_result)

# %%
# calc nll

