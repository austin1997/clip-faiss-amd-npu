# %%
import torch
from PIL import Image
import open_clip

# model_name = "ViT-bigG-14"
# pretrain_dataset = "laion2b_s39b_b160k"
model_name = "ViT-B-32"
pretrain_dataset = "laion2b_s34b_b79k"
model_path = 'D:\\Models\\open-clip\\CLIP-ViT-B-32-laion2B-s34B-b79K\\open_clip_pytorch_model.bin'
# model_name = "convnext_large_d_320"
# pretrain_dataset = "laion2b_s29b_b131k_ft_soup"
# model_path = 'D:\\Models\\open-clip\\CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup\\open_clip_pytorch_model.bin'
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_path, cache_dir="./model_cache")
tokenizer = open_clip.get_tokenizer(model_name)

image = preprocess(Image.open("./demo.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"]).int()
print("text tokens: ", text)
model.eval()

# %%
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    image_features, _, _ = model(image, None)
    _, text_features, _ = model(None, text)
    torch_out = [image_features, text_features]
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs2 = (100.0 * image_features @ text_features.T).softmax(dim=-1)


print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
print("Label probs2:", text_probs2)  # prints: [[1., 0., 0.]]
print("torch_out: ", torch_out)

# %%
# Export image model
import os
import onnx
import onnxsim
def export_model(model, output_path, inputs):
    if os.path.exists(output_path):
        os.remove(output_path)
    torch.onnx.export(model,               # model being run
                inputs,                         # model input (or a tuple for multiple inputs)
                output_path,   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=17,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input.0', 'input.1'],   # the model's input names
                output_names = ['output.0', 'output.1'], # the model's output names
                dynamic_axes={ 
                            # 'input.0' : {0 : 'batch_size'},    # variable length axes
                            # 'input.1' : {0 : 'batch_size'}
                            }
                )
    onnx_model = onnx.load(output_path)
    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_model, output_path)
    # onnx_model = onnx.load()
    # onnx.checker.check_model(onnx_model)
    onnx.checker.check_model(output_path)

onnx_dir = "./onnx/"+model_name+"_"+pretrain_dataset
onnx_path = os.path.join(onnx_dir, "model.onnx")
os.makedirs(onnx_dir, exist_ok=True)
args = (image, text[0].reshape(1, -1))
# onnx_program = torch.onnx.dynamo_export(model, *args)
# onnx_program.save(onnx_path)
# model = torch.jit.script(model)
export_model(model, onnx_path, args)


# %%
# prepare quantize model
import onnx
from onnxruntime.quantization import shape_inference
onnx_dir = "./onnx/"+model_name+"_"+pretrain_dataset
onnx_path = os.path.join(onnx_dir, "model.onnx")
shape_inference.quant_pre_process(onnx_path, os.path.join(onnx_dir, 'model_prepared.onnx'))
onnx_path = os.path.join(onnx_dir, "model_prepared.onnx")
onnx.checker.check_model(onnx_path)

# %%
import onnxruntime
import numpy as np

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image), ort_session.get_inputs()[1].name: to_numpy(text[0].reshape(1, -1))}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(to_numpy(torch_out[1][0]), ort_outs[1][0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")



# %% Export the model dynamo
# with torch.no_grad():
#     import os
#     onnx_dir = "./onnx/"+model_name+"_"+pretrain_dataset + "_dynamo"
#     onnx_path = onnx_dir + "/model.onnx"
#     os.makedirs(onnx_dir, exist_ok=True)
#     args = (image, text)
#     onnx_program = torch.onnx.dynamo_export(model, *args)
#     onnx_program.save(onnx_path)

# %%
