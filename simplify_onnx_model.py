# %%
# load onnx model
import onnx
import os
import numpy as np
from google.protobuf.json_format import MessageToDict
# model_name = "ViT-B/32"
model_name = "RN50"
onnx_dir = "./onnx/" + model_name
onnx_path = os.path.join(onnx_dir, "text_model.onnx")
output_path = os.path.join(onnx_dir, "text_model_reshaped.onnx")
onnx_model = onnx.load(onnx_path)
# onnx.save_model(onnx_model, os.path.join(onnx_dir, "model.textproto"), "textproto")
# %%
# simpify
onnx_graph = onnx_model.graph

print("initializers: ")
initializer_dict: dict[str, onnx.TensorProto] = {}
for initializer in onnx_graph.initializer:
    initializer_dict[initializer.name] = initializer

print("inputs: ")
input_dict : dict[str, onnx.ValueInfoProto] = {}
for input in onnx_graph.input:
    input_dict[input.name] = input
print("value_infos: ")
value_info_dict: dict[str, onnx.ValueInfoProto] = {}
for value in onnx_model.graph.value_info:
    value_info_dict[value.name] = value
print("nodes: ")
node_dict: dict[str, onnx.NodeProto] = {}
input2nodes: dict[str, list] = {}
output2nodes: dict[str, list] = {}
new_nodes = []
for node in onnx_graph.node:
    new_nodes.append(node)
    node_dict[node.name] = node
    for input in node.input:
        tmp = input2nodes.get(input, list())
        tmp.append(node.name)
        input2nodes[input] = tmp
    for output in node.output:
        tmp = output2nodes.get(output, list())
        tmp.append(node.name)
        output2nodes[output] = tmp

def get_tensor_shape(input: str):
    value_info = value_info_dict[input]
    input_shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField('dim_value'):
            input_shape.append(dim.dim_value)
        else:
            input_shape.append(-1)
    return input_shape

new_initializers = []
cnt = 0
for name, node in node_dict.items():
    if node.op_type != "MatMul":
        continue
    if node.input[0] not in value_info_dict.keys() or \
       node.input[1] not in initializer_dict.keys():
        continue
    weight = initializer_dict[node.input[1]]
    weight = onnx.numpy_helper.to_array(weight).copy()
    weight_shape = weight.shape
    input_shape = get_tensor_shape(node.input[0])
    if len(input_shape) == len(weight_shape) or \
       len(weight_shape) != 2 or \
       input_shape[-1] == -1 or \
       input_shape[-1] != weight_shape[0]:
        continue
    output = node.output[0]
    output_users = input2nodes[output]
    if len(output_users) > 1:
        continue
    user = node_dict[output_users[0]]
    original_input_name = node.input[0]
    if user.op_type != "Add":
        original_output_name = node.output[0]
        original_output_shape = get_tensor_shape(node.output[0])
        bias_initializer = None
    else:
        original_output_name = user.output[0]
        original_output_shape = get_tensor_shape(user.output[0])
        if user.input[0] in initializer_dict.keys():
            bias_initializer = user.input[0]
        elif user.input[1] in initializer_dict.keys():
            bias_initializer = user.input[1]
        else:
            continue
    print(name)

    # create reshape
    reshape_name = "Gemm_reshape_before_" + str(cnt)
    reshape_out_name = reshape_name + "_out"
    reshape_init_name = reshape_name + "_initializer"
    tensor_array = np.ones((2))
    for i in range(0, len(input_shape) - 1):
        tensor_array[0] *= input_shape[i]
    if tensor_array[0] < 0:
        tensor_array[0] = -1
    tensor_array[1] = input_shape[-1]
    new_initializer_tensor = onnx.helper.make_tensor(
        name=reshape_init_name,
        data_type=onnx.TensorProto.INT64,
        dims=[2],
        vals=tensor_array.astype(np.int64).flatten().tolist())
    new_node = onnx.helper.make_node("Reshape", [original_input_name, reshape_init_name], [reshape_out_name], name=reshape_name)
    node.input[0] = reshape_out_name
    index = new_nodes.index(node)
    new_nodes.insert(index, new_node)
    new_initializers.append(new_initializer_tensor)
    # print(new_node)
    # create gemm
    new_nodes.remove(node)
    new_nodes.remove(user)
    gemm_name = "fused_gemm_" + str(cnt)
    gemm_out_name = gemm_name + "_out"
    if bias_initializer != None:
        new_node = onnx.helper.make_node("Gemm", [reshape_out_name, node.input[1], bias_initializer], [gemm_out_name], name=gemm_name)
    else:
        new_node = onnx.helper.make_node("Gemm", [reshape_out_name, node.input[1]], [gemm_out_name], name=gemm_name)
    new_attr = onnx.helper.make_attribute("alpha", 1.0)
    new_node.attribute.append(new_attr)
    new_attr = onnx.helper.make_attribute("beta", 1.0)
    new_node.attribute.append(new_attr)
    new_attr = onnx.helper.make_attribute("transB", 0)
    new_node.attribute.append(new_attr)
    new_nodes.insert(index + 1, new_node)
    # create reshape
    reshape_name = "Gemm_reshape_after_" + str(cnt)
    reshape_out_name = reshape_name + "_out"
    reshape_init_name = reshape_name + "_initializer"
    new_initializer_tensor = onnx.helper.make_tensor(
        name=reshape_init_name,
        data_type=onnx.TensorProto.INT64,
        dims=[len(original_output_shape)],
        vals=np.array(original_output_shape).astype(np.int64).flatten().tolist())
    new_node = onnx.helper.make_node("Reshape", [gemm_out_name, reshape_init_name], [original_output_name], name=reshape_name)
    new_nodes.insert(index + 2, new_node)
    new_initializers.append(new_initializer_tensor)
    cnt += 1

new_initializers.extend(onnx_graph.initializer)
graph_def = onnx.helper.make_graph(
        nodes=new_nodes,
        name=onnx_graph.name + "_reshaped",
        inputs=onnx_graph.input,  # Graph input
        outputs=onnx_graph.output,  # Graph output
        initializer=new_initializers,
    )
model_def = onnx.helper.make_model(graph_def, producer_name=onnx_model.producer_name)
model_def.opset_import[0].version = onnx_model.opset_import[0].version
model_def.ir_version = onnx_model.ir_version
# onnx.save_model(model_def, os.path.join(onnx_dir, "model_reshaped.textproto"), "textproto")
model_def = onnx.shape_inference.infer_shapes(model_def)
onnx.checker.check_model(model_def)
# onnx.save(model_def, os.path.join(onnx_dir, "image_model_1_reshaped.onnx"))
import onnxsim
model_def, check = onnxsim.simplify(model_def)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_def, output_path)
# %%
# simplify with onnxruntime
import onnxruntime as rt

onnx_model = onnx.load(os.path.join(onnx_dir, "image_model_1_reshaped.onnx"))
# model_def = onnx.version_converter.convert_version(model_def, 8)
sess_options = rt.SessionOptions()

# Set graph optimization level
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# To enable model serialization after graph optimization set this
sess_options.optimized_model_filepath = os.path.join(onnx_dir, "image_model_1_opt.onnx")

session = rt.InferenceSession(onnx_model.SerializeToString(), sess_options, providers=["CPUExecutionProvider"])

# %%
