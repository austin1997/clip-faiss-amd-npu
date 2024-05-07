# %%
# split and simplify model
import onnx
import onnxsim
import os

model_name = "ViT-B-32"
pretrain_dataset = "laion2b_s34b_b79k"
# model_name = "convnext_large_d_320"
# pretrain_dataset = "laion2b_s29b_b131k_ft_soup"

def remove_input_with_name(model, input_name: str):
    for node in model.graph.input:
        if node.name == input_name:
            target = node
            break
    model.graph.input.remove(target)
    model, check = onnxsim.simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    return model

# text model
onnx_dir = "./onnx/"+model_name+"_"+pretrain_dataset
onnx_path = os.path.join(onnx_dir, "model_prepared.onnx")
onnx_model = onnx.load(onnx_path)
text_model, check = onnxsim.simplify(onnx_model, unused_output=['output.0'])
text_model = remove_input_with_name(text_model, 'input.0')
filepath = os.path.join(onnx_dir, "text_model.onnx")
onnx.save(text_model, filepath)

# image model
onnx_model = onnx.load(onnx_path)
image_model, check = onnxsim.simplify(onnx_model, unused_output=['output.1'])
assert check, "Simplified ONNX model could not be validated"
image_model = remove_input_with_name(image_model, 'input.1')
filepath = os.path.join(onnx_dir, "image_model.onnx")
onnx.save(image_model, filepath)
# %%
