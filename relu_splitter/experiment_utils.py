from relu_splitter.anywhere import ReluSplitter_Anywhere
from relu_splitter.model import WarppedOnnxModel


def get_layer_sizes(type, onnx_path):
    if type == "fc":
        node_type = "gemm"
    elif type == "conv":
        node_type = "conv"
    else:
        raise ValueError(f"Unsupported layer type: {type}")
    

    model = WarppedOnnxModel.load(onnx_path)
    splitable_nodes = ReluSplitter_Anywhere.get_splittable_nodes_cls(model)

    idx_n_size = {}
    # keep the idx and layer size of each node that matches the type
    for idx, node in enumerate(splitable_nodes):
        if node.op_type.lower() == node_type:
            if node_type == "gemm":
                w, b = model.get_gemm_wb(node)
                layer_size = b.shape[0]
            elif node_type == "conv":
                w, b = model.get_conv_wb(node)
                layer_size = b.shape[0]
            idx_n_size[idx] = layer_size
    return idx_n_size
