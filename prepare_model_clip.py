# %%
# load model
import torch
from PIL import Image
import clip

print(clip.available_models())
device = "cuda" if torch.cuda.is_available() else "cpu"
# model_name = "ViT-B/32"
model_name = "RN50"
model, preprocess = clip.load(model_name, device=device, download_root="./Models")
model.eval()
forword_bk = model.forward

import numpy as np
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# %%
# get features
image = preprocess(Image.open("./demo.png")).unsqueeze(0).to(device)
text = ["a diagram", "a dog", "a cat", "a koala"]
text = clip.tokenize(text).to(device).int()

# %%
# Export the splited model
import os
import onnx
import onnxsim
def export_onnx_model(model, input, dir_path, filename, dynamic = False):
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    if dynamic:
        dynamic_axes = {
                        'input.0' : {0 : 'batch_size'},
                        # 'output.0' : {0 : 'batch_size'}
                       }
    else:
        dynamic_axes = None
    torch.onnx.export(model,               # model being run
                input,                         # model input (or a tuple for multiple inputs)
                filepath,   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=17,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input.0'],   # the model's input names
                output_names = ['output.0'], # the model's output names
                dynamic_axes= dynamic_axes  # variable length axes
                )
    onnx_model = onnx.load(filepath)
    onnx.checker.check_model(onnx_model)
    # onnx.checker.check_model(onnx_path)
    # convert model
    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_model, filepath)

# onnx_program = torch.onnx.dynamo_export(model, *args)
# onnx_program.save(onnx_path)
# model = torch.jit.script(model)
text = text[0].reshape(1, -1)
onnx_dir = "./output_models/" + model_name
model.forward = forword_bk
export_onnx_model(model, (image, text), onnx_dir, 'model.onnx', True)
model.forward = model.encode_image
export_onnx_model(model, (image,), onnx_dir, 'image_model.onnx', True if 'RN50' in model_name else False)
model.forward = model.encode_text
export_onnx_model(model, (text,), onnx_dir, 'text_model.onnx')

# %%
# run onnx model
import onnxruntime

def run_onnx_session(session, input):
    ort_inputs = {session.get_inputs()[0].name: to_numpy(input)}
    ort_outs = session.run(None, ort_inputs)
    return ort_outs

def run_onnx_session2(session, input0, input1):
    ort_inputs = {session.get_inputs()[0].name: to_numpy(input0), session.get_inputs()[1].name: to_numpy(input1)}
    ort_outs = session.run(None, ort_inputs)
    return ort_outs

text = text[0].reshape(1, -1)
onnx_path = os.path.join(onnx_dir, "image_model.onnx")
ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
# compute ONNX Runtime output prediction
ort_image_out = run_onnx_session(ort_session, image)

onnx_path = os.path.join(onnx_dir, "text_model.onnx")
ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
ort_text_out = run_onnx_session(ort_session, text)
ort_outs = [ort_image_out[0], ort_text_out[0]]

onnx_path = os.path.join(onnx_dir, "model.onnx")
ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
ort_model_outs = run_onnx_session2(ort_session, image, text)

# %%
# compare ONNX Runtime results
np.testing.assert_allclose(to_numpy(model.encode_image(image)), ort_outs[0], rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(to_numpy(model.encode_text(text)), ort_outs[1], rtol=1e-03, atol=1e-05)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# %% split onnx model by name
import onnx
# split_tensor_name = '/visual/attnpool/Add_output_0'
split_tensor_name = '/visual/layer4/layer4.2/relu3/Relu_output_0'
onnx_path = os.path.join(onnx_dir, "image_model.onnx")
output_path = os.path.join(onnx_dir, "image_model_0.onnx")
onnx.utils.extract_model(onnx_path, output_path, ['input.0'], [split_tensor_name])
output_path = os.path.join(onnx_dir, "image_model_1.onnx")
onnx.utils.extract_model(onnx_path, output_path, [split_tensor_name], ['output.0'])

# %%
# prepare calibration dataset
import onnxruntime
import onnxruntime.quantization
import pandas as pd
import requests
import pickle
dataset_path = './Datasets/lvis_caption_url.parquet'
datasets = pd.read_parquet(dataset_path)
datasets = datasets.head(1000)

class DataReader(onnxruntime.quantization.CalibrationDataReader):
    def __init__(self, dataframe: pd.DataFrame, input_name: str, is_image: bool, len = 1000, resolution = None):
        self.features = []
        if is_image:
            cache_path = './Datasets/clip-image.pkl'
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    images = pickle.load(f)
            else:
                images = []
                for data in dataframe['url']:
                    images.append(Image.open(requests.get(data, stream=True).raw))
                with open(cache_path, 'wb') as f:
                    pickle.dump(images, f, pickle.HIGHEST_PROTOCOL)
            images = images[:len]
            for image in images:
                self.features.append(to_numpy(preprocess(image).unsqueeze(0)))
            del images
        else:
            for data in dataframe['caption']:
                try:
                    tmp = to_numpy(clip.tokenize(data).int())
                except RuntimeError:
                    pass
                # tmp[tmp == 0] = eps
                self.features.append(tmp.reshape(1, -1))
        self.enum_data = None
        self.input_name = input_name
        self.cnt = 0

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: data} for data in self.features]
            )
        # if self.cnt % 10 == 0:
        #     print("cnt: ", self.cnt)
        self.cnt += 1
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

# %%
# quantize image model
import vai_q_onnx

data_reader = DataReader(datasets, 'input.0', True, 150, 224)
onnx_path = os.path.join(onnx_dir, "image_model.onnx")
output_path = os.path.join(onnx_dir, "image_model_quantized.onnx")
onnxruntime.quantization.shape_inference.quant_pre_process(onnx_path, output_model_path=output_path)
# CNN on NPU
vai_q_onnx.quantize_static(
   output_path,
   output_path,
   data_reader,
   quant_format=vai_q_onnx.QuantFormat.QDQ,
   calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
   activation_type=vai_q_onnx.QuantType.QUInt8,
   weight_type=vai_q_onnx.QuantType.QInt8,
   enable_ipu_cnn=True,
   include_cle=True,
   extra_options={
       'ActivationSymmetric':True,
       'ReplaceClip6Relu': True,
       'CLESteps': 1,
       'CLEScaleAppendBias': True,
                  },
)
print("Quantize Resnet50 & export ONNX done")
