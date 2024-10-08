from faiss import write_index
from PIL import Image
from tqdm import tqdm

import argparse
import clip
import faiss
import json
import numpy as np
import os
import torch
import onnxruntime

model_name = "RN50"
onnx_dir = "./output_models/"+model_name
onnx_path = os.path.join(onnx_dir, "image_model_quantized.onnx")
config_path = '.\\vaip_config.json'
session0 = onnxruntime.InferenceSession(
            onnx_path,
            providers=["VitisAIExecutionProvider"],
            provider_options=[{"config_file":config_path}])

onnx_path = os.path.join(onnx_dir, "image_model_1.onnx")
sess_options = onnxruntime.SessionOptions()

# Set graph optimization level
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
session1 = onnxruntime.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def eval(image_input):
    # Load inputs and do preprocessing by input_shape
    input_name = session0.get_inputs()[0].name
    input_shape = session0.get_inputs()[0].shape
    for i in range(0, len(input_shape)):
        if not isinstance(input_shape[i], int):
            input_shape[i] = 1
    image_input = np.reshape(image_input, input_shape)
    npu_qresult = session0.run([], {input_name: image_input})

    input_name = session1.get_inputs()[0].name
    npu_qresult = session1.run([], {input_name: npu_qresult[0]})
    return npu_qresult[0]

def get_image_files(dir_path: str):
    result = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                result.append(os.path.join(root, file))
    return result

def index(image_dir_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, preprocess = clip.load(model_name, device=device, download_root='./Models')

    image_paths = []
    img_dir_path = image_dir_path
    image_features = []
    for img_file in tqdm(get_image_files(img_dir_path)):
        image = Image.open(img_file).convert("RGB")
        image = to_numpy(preprocess(image))
        image_paths.append(img_file)
        image_features.append(eval(image)[0])

    image_features = torch.tensor(np.stack(image_features)).to(device)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().numpy()
    index = faiss.IndexFlatIP(image_features.shape[1])
    index.add(image_features)
    write_index(index, "static/index.faiss")

    with open("static/image_paths.json", "w") as f:
        json.dump(image_paths, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir_path", type=str, default="static/data/images")
    args = parser.parse_args()
    index(args.image_dir_path)
