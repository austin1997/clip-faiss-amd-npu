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
onnx_dir = "./onnx/"+model_name
onnx_path = os.path.join(onnx_dir, "image_model_0_quantized.onnx")
config_path = 'C:\\Users\\austi\\Downloads\\ryzen-ai-sw-1.1\\ryzen-ai-sw-1.1\\voe-4.0-win_amd64\\vaip_config.json'
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
    # print(image_input.shape)
    input_name = session0.get_inputs()[0].name
    input_shape = session0.get_inputs()[0].shape
    for i in range(0, len(input_shape)):
        if not isinstance(input_shape[i], int):
            input_shape[i] = 1
    image_input = np.reshape(image_input, input_shape)
    npu_qresult = session0.run([], {input_name: image_input})
    # print("Quantized: ", npu_qresult)

    input_name = session1.get_inputs()[0].name
    npu_qresult = session1.run([], {input_name: npu_qresult[0]})
    # image_features = torch.from_numpy(npu_qresult[0])
    # print("Quantized: ", npu_qresult)
    return npu_qresult[0]

def index(image_dir_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load(model_name, device=device, download_root='D:\\Models\\CLIP\\')
    model.eval()

    images = []
    image_paths = []
    img_dir_path = image_dir_path
    image_features = []
    for animal_name in sorted(os.listdir(img_dir_path)):
        print(animal_name)
        if not os.path.isdir(os.path.join(img_dir_path, animal_name)):
            continue
        for img_file in tqdm(os.listdir(os.path.join(img_dir_path, animal_name))):
            if not img_file.endswith(".jpg"):
                continue
            image = Image.open(os.path.join(img_dir_path, animal_name, img_file)).convert("RGB")
            # images.append(preprocess(image))
            image = to_numpy(preprocess(image))
            image_paths.append(os.path.join(img_dir_path, animal_name, img_file))
            image_features.append(eval(image)[0])

    # image_input = torch.tensor(np.stack(images)).to(device)
    image_features = torch.tensor(np.stack(image_features)).to(device)
    # with torch.no_grad():
    #     image_features = model.encode_image(image_input).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().numpy()
    print(image_features.shape)
    index = faiss.IndexFlatIP(image_features.shape[1])
    index.add(image_features)
    write_index(index, "static/index.faiss")

    with open("static/image_paths.json", "w") as f:
        json.dump(image_paths, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_dir_path", type=str, default="static/data/images")
    parser.add_argument("--image_dir_path", type=str, default="D:\\Datasets\\ye-pop\\images\\chunk_1")
    args = parser.parse_args()
    index(args.image_dir_path)
