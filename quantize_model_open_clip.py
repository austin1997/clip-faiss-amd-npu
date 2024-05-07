# %%
from PIL import Image
import open_clip
import os
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
import gc
gc.collect()

# %%
# prepare calibration data
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
import onnxruntime
import onnxruntime.quantization
import pandas as pd
import requests
import numpy as np
import os
import pickle
dataset_path = 'D:\\Datasets\\220k-GPT4Vision-captions-from-LIVIS\\lvis_caption_url.parquet'
datasets = pd.read_parquet(dataset_path)
datasets = datasets.head(1000)

class DataReader(onnxruntime.quantization.CalibrationDataReader):
    def __init__(self, dataframe: pd.DataFrame, input_name: str, is_image: bool, resolution = None):
        self.features = []
        if is_image:
            cache_path = 'D:\\Datasets\\clip-image.pkl'
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    images = pickle.load(f)
            else:
                images = []
                for data in dataframe['url']:
                    images.append(Image.open(requests.get(data, stream=True).raw))
                with open(cache_path, 'wb') as f:
                    pickle.dump(images, f, pickle.HIGHEST_PROTOCOL)
            for image in images:
                self.features.append(to_numpy(preprocess(image).unsqueeze(0)))
            del images
        else:
            for data in dataframe['caption']:
                tmp = to_numpy(tokenizer(data))
                # tmp[tmp == 0] = eps
                self.features.append(tmp.reshape(1, -1))
        self.enum_data = None
        self.input_name = input_name

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: data} for data in self.features]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None
        
# data_reader = DataReader(datasets, 'input.1', False)

# %%
# quantize image model
import vai_q_onnx

data_reader = DataReader(datasets, 'input.0', True, 224)
# data_reader = DataReader(datasets, 'input.0', True, 320)
onnx_dir = "./onnx/"+model_name+"_"+pretrain_dataset
onnx_path = os.path.join(onnx_dir, "image_model.onnx")
output_path = os.path.join(onnx_dir, "image_model_quantized.onnx")
# CNN on NPU
# vai_q_onnx.quantize_static(
#    onnx_path,
#    output_path,
#    None,
#    quant_format=vai_q_onnx.QuantFormat.QDQ,
#    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
#    activation_type=vai_q_onnx.QuantType.QUInt8,
#    weight_type=vai_q_onnx.QuantType.QInt8,
#    enable_dpu=True,
#    extra_options={'ActivationSymmetric':True},
#    convert_nchw_to_nhwc=True
# )

# Transformer on NPU
vai_q_onnx.quantize_static(
   onnx_path,
   output_path,
   data_reader,
   quant_format=vai_q_onnx.QuantFormat.QDQ,
   calibrate_method=vai_q_onnx.CalibrationMethod.MinMax,
   activation_type=vai_q_onnx.QuantType.QInt8,
   weight_type=vai_q_onnx.QuantType.QInt8,
   enable_dpu=True,
)

# %%
# quantize text model
import vai_q_onnx

data_reader = DataReader(datasets, 'input.1', False)
onnx_dir = "./onnx/"+model_name+"_"+pretrain_dataset
onnx_path = os.path.join(onnx_dir, "text_model.onnx")
output_path = os.path.join(onnx_dir, "text_model_quantized.onnx")

exclude_node_list = []

vai_q_onnx.quantize_static(
   onnx_path,
   output_path,
   data_reader,
   quant_format=vai_q_onnx.QuantFormat.QDQ,
   calibrate_method=vai_q_onnx.CalibrationMethod.MinMax,
   activation_type=vai_q_onnx.QuantType.QInt8,
   weight_type=vai_q_onnx.QuantType.QInt8,
)
