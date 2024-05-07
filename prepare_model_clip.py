# %%
# load model
import torch
from PIL import Image
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B/32"
model, preprocess = clip.load(model_name, device=device, download_root="D:\\Models\\CLIP\\")
model.eval()

# %%
# get features
image = preprocess(Image.open("./demo.png")).unsqueeze(0).to(device)
text = ["a diagram", "a dog", "a cat"]
text = clip.tokenize(text).to(device)

# %%
# eval model
forword_bk = model.forward
with torch.no_grad():
    model_out = model(image, text)
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    torch_out = [image_features, text_features]
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    setattr(model, 'forward', model.encode_image)
    image_features = model(image)
    setattr(model, 'forward', model.encode_text)
    text_features = model(text)
    torch_out2 = [image_features, text_features]
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs2 = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
print("Label probs2:", text_probs2)  # prints: [[1., 0., 0.]]
print("torch_out: ", torch_out)

import numpy as np
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
np.testing.assert_allclose(to_numpy(torch_out[0]), to_numpy(torch_out2[0]), rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(to_numpy(torch_out[1]), to_numpy(torch_out2[1]), rtol=1e-03, atol=1e-05)


# %%
# Export the image model
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
                dynamic_axes={'input.0' : {0 : 'batch_size'}}# variable length axes
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
onnx_dir = "./onnx/" + model_name
model.forward = forword_bk
export_onnx_model(model, (image, text), onnx_dir, 'model.onnx')
setattr(model, 'forward', model.encode_image)
export_onnx_model(model, (image,), onnx_dir, 'image_model.onnx')
setattr(model, 'forward', model.encode_text)
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

onnx_dir = onnx_dir
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
np.testing.assert_allclose(to_numpy(model_out[0]), ort_model_outs[0], rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(to_numpy(model_out[1]), ort_model_outs[1], rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(to_numpy(torch_out[1]), ort_outs[1], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

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
