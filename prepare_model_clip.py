# %%
# load model
import torch
from PIL import Image
import clip

print(clip.available_models())
device = "cuda" if torch.cuda.is_available() else "cpu"
# model_name = "ViT-B/32"
model_name = "RN50"
model, preprocess = clip.load(model_name, device=device, download_root="D:\\Models\\CLIP\\")
model.eval()
print(model)
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
# eval model
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    print('image_features: ', image_features)
    torch_out = [to_numpy(image_features), to_numpy(text_features)]
    print('torch_out[0]: ', torch_out[0])
    model.forward = forword_bk
    model_out = model(image, text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    model.forward = model.encode_image
    image_features = model(image)
    model.forward = model.encode_text
    text_features = model(text)
    torch_out2 = [to_numpy(image_features), to_numpy(text_features)]
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs2 = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
print("Label probs2:", text_probs2)  # prints: [[1., 0., 0.]]
print("torch_out: ", torch_out)

np.testing.assert_allclose(torch_out[0], torch_out2[0], rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(torch_out[1], torch_out2[1], rtol=1e-03, atol=1e-05)


# %%
# Export the splited model
import os
import onnx
import onnxsim
def export_onnx_model(model, input, dir_path, filename):
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    torch.onnx.export(model,               # model being run
                input,                         # model input (or a tuple for multiple inputs)
                filepath,   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=17,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input.0'],   # the model's input names
                output_names = ['output.0'], # the model's output names
                # dynamic_axes={
                #     'input.0' : {0 : 'batch_size'},
                #     'output.0' : {0 : 'batch_size'}
                # }# variable length axes
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
onnx_dir = "./onnx/" + model_name
model.forward = forword_bk
export_onnx_model(model, (image, text), onnx_dir, 'model.onnx')
model.forward = model.encode_image
export_onnx_model(model, (image,), onnx_dir, 'image_model.onnx')
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

onnx_dir = "./onnx/" + model_name
text = text[0].reshape(1, -1)
onnx_path = os.path.join(onnx_dir, "image_model.onnx")
ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
# compute ONNX Runtime output prediction
ort_image_out = run_onnx_session(ort_session, image)

onnx_path = os.path.join(onnx_dir, "text_model.onnx")
ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
ort_text_out = run_onnx_session(ort_session, text)
ort_outs = [ort_image_out[0], ort_text_out[0]]
print(ort_outs)

onnx_path = os.path.join(onnx_dir, "model.onnx")
ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
ort_model_outs = run_onnx_session2(ort_session, image, text)


# %%
# compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(model_out[0]), ort_model_outs[0], rtol=1e-03, atol=1e-05)
# np.testing.assert_allclose(to_numpy(model_out[1]), ort_model_outs[1], rtol=1e-03, atol=1e-05)
# np.testing.assert_allclose(torch_out[0], ort_outs[0], rtol=1e-03, atol=1e-05)
model.forward = forword_bk
np.testing.assert_allclose(to_numpy(model.encode_image(image)), ort_outs[0], rtol=1e-03, atol=1e-05)
with torch.no_grad():
    np.testing.assert_allclose(to_numpy(model.encode_image(image)), ort_outs[0], rtol=1e-03, atol=1e-05)
# np.testing.assert_allclose(to_numpy(torch_out[1]), ort_outs[1], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# %% split onnx model by name
import onnx
split_tensor_name = '/visual/attnpool/Add_output_0'
onnx_dir = "./onnx/" + model_name
onnx_path = os.path.join(onnx_dir, "image_model.onnx")
output_path = os.path.join(onnx_dir, "image_model_0.onnx")
onnx.utils.extract_model(onnx_path, output_path, ['input.0'], [split_tensor_name])
output_path = os.path.join(onnx_dir, "image_model_1.onnx")
onnx.utils.extract_model(onnx_path, output_path, [split_tensor_name], ['output.0'])

# %% Export the model dynamo
model.forward = model.encode_image
compiled_image_model = torch.compile(model)
onnx_program = torch.onnx.dynamo_export(compiled_image_model, image)
onnx_program.save(os.path.join(onnx_dir, 'image_model_dynamo.onnx'))
with torch.no_grad():
    import os
    onnx_dir = "./onnx/"+model_name+"_"+pretrain_dataset + "_dynamo"
    onnx_path = onnx_dir + "/model.onnx"
    os.makedirs(onnx_dir, exist_ok=True)
    args = (image, text)
    onnx_program = torch.onnx.dynamo_export(model, *args)
    onnx_program.save(onnx_path)

# %%
# prepare calibration dataset
import onnxruntime
import onnxruntime.quantization
import pandas as pd
import requests
import pickle
dataset_path = 'D:\\Datasets\\220k-GPT4Vision-captions-from-LIVIS\\lvis_caption_url.parquet'
datasets = pd.read_parquet(dataset_path)
datasets = datasets.head(1000)

class DataReader(onnxruntime.quantization.CalibrationDataReader):
    def __init__(self, dataframe: pd.DataFrame, input_name: str, is_image: bool, len = 1000, resolution = None):
        self.features = []
        if is_image:
            cache_path = 'D:\\Datasets\\220k-GPT4Vision-captions-from-LIVIS\\clip-image.pkl'
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

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: data} for data in self.features]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

# %%
# quantize image model
import vai_q_onnx

data_reader = DataReader(datasets, 'input.0', True, 200, 224)
onnx_dir = "./onnx/" + model_name
onnx_path = os.path.join(onnx_dir, "image_model_0.onnx")
output_path = os.path.join(onnx_dir, "image_model_0_quantized.onnx")
# CNN on NPU
vai_q_onnx.quantize_static(
   onnx_path,
   output_path,
   data_reader,
   quant_format=vai_q_onnx.QuantFormat.QDQ,
   calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
   activation_type=vai_q_onnx.QuantType.QUInt8,
   weight_type=vai_q_onnx.QuantType.QInt8,
   enable_ipu_cnn=True,
   extra_options={'ActivationSymmetric':True},
#    convert_nchw_to_nhwc=True
)

# %%
# quantize second part
import vai_q_onnx
# data_reader = DataReader(datasets, split_tensor_name, False, 200, 224)
onnx_dir = "./onnx/" + model_name
onnx_path = os.path.join(onnx_dir, "image_model_1_reshaped.onnx")
output_path = os.path.join(onnx_dir, "image_model_1_quantized.onnx")
# onnx_path = os.path.join(onnx_dir, "image_model.onnx")
# output_path = os.path.join(onnx_dir, "image_model_quantized.onnx")
# Transformer on NPU
vai_q_onnx.quantize_static(
   onnx_path,
   output_path,
   None,
   quant_format=vai_q_onnx.QuantFormat.QDQ,
   calibrate_method=vai_q_onnx.CalibrationMethod.MinMax,
   activation_type=vai_q_onnx.QuantType.QInt8,
   weight_type=vai_q_onnx.QuantType.QInt8,
   enable_ipu_transformer=True,
)

# %%
# merge splited models
import onnx
onnx_dir = "./onnx/" + model_name
input_path1 = os.path.join(onnx_dir, "image_model_0_quantized.onnx")
input_path2 = os.path.join(onnx_dir, "image_model_1_quantized.onnx")
model1 = onnx.load(input_path1)
model2 = onnx.load(input_path2)
combined_model = onnx.compose.merge_models(
    model1, model2,
    io_map=[(split_tensor_name, split_tensor_name)]
)

# %%
# quantize text model
import vai_q_onnx

data_reader = DataReader(datasets, 'input.0', False)
onnx_dir = "./onnx/" + model_name
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
   enable_ipu_transformer=True
)

# %%
# run quantized image model on NPU
import onnxruntime
import os
import numpy as np
# Add user imports
# ...

# Load inputs and perform preprocessing
# ...

# Create an inference session using the Vitis AI execution provider

onnx_dir = "./onnx/" + model_name
onnx_path = os.path.join(onnx_dir, "image_model_0_quantized.onnx")
# onnx_path = os.path.join(onnx_dir, "image_model_0_quantized.onnx")
# onnx_path = "D:\\git_repo\\clip-faiss-amd-npu\\quantize_result\\CLIP_int.onnx"
config_path = 'C:\\Users\\austi\\Downloads\\ryzen-ai-sw-1.1\\ryzen-ai-sw-1.1\\voe-4.0-win_amd64\\vaip_config.json'
session = onnxruntime.InferenceSession(
               onnx_path,
               providers=["VitisAIExecutionProvider"],
               provider_options=[{"config_file":config_path}])

input_shape = session.get_inputs()[0].shape
input_name = session.get_inputs()[0].name

# Load inputs and do preprocessing by input_shape
# input_data = np.transpose(to_numpy(image), (0, 2, 3, 1))
input_data = to_numpy(image)
npu_qresult = session.run([], {input_name: input_data})

# %%
# run second part
# onnx_path = os.path.join(onnx_dir, "image_model_1.onnx")
# session = onnxruntime.InferenceSession(
#                onnx_path,
#                providers=["CPUExecutionProvider"])
onnx_path = os.path.join(onnx_dir, "image_model_1_reshaped.onnx")
session = onnxruntime.InferenceSession(
               onnx_path,
               providers=["VitisAIExecutionProvider"],
               provider_options=[{"config_file":config_path}])
input_shape = session.get_inputs()[0].shape
input_name = session.get_inputs()[0].name
# npu_qresult = session.run([], {input_name: npu_qresult[0]})
npu_qresult = session.run([], {input_name: np.random.rand(*input_shape).astype(np.float32)})
original_result = model.encode_image(image)
print("Quantized: ", npu_qresult)
print("original: ", original_result)

# %%
# calc loss
loss_fct = torch.nn.CrossEntropyLoss()
loss = loss_fct(
    torch.tensor(npu_qresult[0]),
    original_result 
)
print("loss: ", loss.float())
image_features = torch.tensor(npu_qresult[0])
text_features = model.encode_text(text)
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
print(text_probs)
# %%
