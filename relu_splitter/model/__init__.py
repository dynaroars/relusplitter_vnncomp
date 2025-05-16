import logging
from pathlib import Path
from typing import Tuple
from functools import reduce
from copy import copy, deepcopy

import onnx
import torch

from onnx import helper, numpy_helper, version_converter
from onnxruntime import InferenceSession
from onnx2pytorch import ConvertModel
from auto_LiRPA import BoundedModule, BoundedTensor

from ..utils.onnx_utils import truncate_onnx_model, compute_model_bound
from ..utils.misc import get_random_id

SINGLE_INPUT_OPS = ["Relu", "MatMul", "Add"]

custom_quirks = {
    'Reshape': {
        'fix_batch_size': False
    },
    'Transpose': {
        'merge_batch_size_with_channel': True,
        'remove_gdvb_transpose': True,
    },
    'Softmax' :{
        'skip_last_layer': True
    },
    'Squeeze' :{
        'skip_last_layer': True
    },
    'Conv' :{
        'merge_batch_norm': True
    },
}

default_logger = logging.getLogger(__name__)

class WarppedOnnxModel():
    def __init__(self, model: onnx.ModelProto, force_rename = False, logger=default_logger)-> None:
        self.logger = logger
        self.force_rename = force_rename

        self._sanity_check(model)

        self._model_ = model    # protected, immutable
        self.op_set = model.opset_import
        self._initializers = [initializer for initializer in model.graph.initializer]
        self._initializers_mapping = {initializer.name: initializer for initializer in model.graph.initializer}
        self._nodes = [node for node in model.graph.node]
        self._nodes_mapping = {node.name: node for node in model.graph.node}
        # self.node_accepts_input = {node.input[0]: node for node in model.graph.node}
        self._node_produce_output = {output: node for node in model.graph.node for output in node.output}
    
    def _sanity_check(self, model: onnx.ModelProto):
        # check for duplicate node name or no node name
        node_names = [node.name for node in model.graph.node]
        has_duplicate_node_name = (len(node_names) != len(set(node_names)))
        has_node_w_empty_name    = any((node.name == "") for node in model.graph.node)
        if has_duplicate_node_name or has_node_w_empty_name or self.force_rename:
            for i, node in enumerate(model.graph.node):
                node.name = f"{node.op_type}_{i}"
            self.logger.info(f"Renamed nodes...")
        # for node in model.graph.node:
            # each node only produce one output
            # assert len(node.output) == 1, f"Node {node.name} produces more than one output"
            # allowed op_types
            # ...
        # remove initializer from model input
        for initializer in model.graph.initializer:
            for i, input_tensor in enumerate(model.graph.input):
                if input_tensor.name == initializer.name:
                    self.logger.info(f"Removing initializer {initializer.name} from model input")
                    del model.graph.input[i]
                    break

    def info(self):
        s = f"Model info:\n"
        s += f"\tModel input: {self.input_shapes}\n"
        s += f"\tModel output: {self._model.graph.output[0].name}\n"
        s += f"\tModel opset: {self.op_set}\n"
        s += "====================\n"
        for node in self._nodes:
            s += f"\t\tNode name: {node.name}, Node type: {node.op_type}\n"
            s += f"\t\t\tInputs: {node.input}\n"
            s += f"\t\t\tOutputs: {node.output}\n"
        self.logger.info(s)
        return s

    def gen_node_name(self, prefix: str):
        name = f"{prefix}_{get_random_id()}"
        while self.has_node(name):
            name = f"{prefix}_{get_random_id()}"
        return name
    
    def has_node(self, node_name: str):
        return node_name in self._nodes_mapping.keys()

    def gen_tensor_name(self, prefix: str):
        name = f"{prefix}_{get_random_id()}"
        while self.has_tensor(name):
            name = f"{prefix}_{get_random_id()}"
        return name

    def has_tensor(self, tensor_name: str):
        if not hasattr(self, "_tensor_names"):
            self._tensor_names = set()
            for node in self._nodes:
                self._tensor_names.update(node.input)
                self._tensor_names.update(node.output)
        return tensor_name in self._tensor_names

    def has_initializer(self, initializer_name: str):
        return initializer_name in self._initializers_mapping.keys()
    
    def get_node_initializers(self, node):
        return [i for i in node.input if self.has_initializer(i)]
    
    def get_node_inputs_no_initializers(self, node):
        return [i for i in node.input if not self.has_initializer(i)]

    @property
    def _model(self):
        return deepcopy(self._model_)

    @property
    def nodes(self):
        return copy(self._nodes)
    
    @property
    def num_inputs(self):
        count = 0
        for input_tensor in self.input_shapes.values():
            count += reduce(lambda x, y: x*y, input_tensor)
        return count

    @property
    def input_shapes(self):
        try:
            input_shapes = {}
            for input_tensor in self._model.graph.input:
                shape = []
                for dim in input_tensor.type.tensor_type.shape.dim:
                    if dim.dim_value == 0:
                        self.logger.info("skip the input shape with 0 dim_value")
                        continue
                    shape.append(dim.dim_value)
                input_shapes[input_tensor.name] = shape
            return input_shapes
        except:
            self.logger.warning("Error retrieving the input shape, Infering the input shape...")
            return self._infer_input_shapes()
        
    def _infer_input_shapes(self):
        input_shapes = {}
        for input_tensor in self._model.graph.input:
            shape = []
            for dim in input_tensor.type.tensor_type.shape.dim:
                shape.append(dim.dim_value)
            input_shapes[input_tensor.name] = shape
        return input_shapes

    def get_prior_node(self, node):
        assert node.op_type in SINGLE_INPUT_OPS
        assert len(node.input) == 1
        return self._node_produce_output[node.input[0]]

    def get_bound_of(self, input_bound:BoundedTensor, tensor_name: str, method: str = "forward+backward") -> Tuple[torch.Tensor, torch.Tensor]:
        # trunated_model = truncate_onnx_model(self._model, tensor_name)
        # return compute_model_bound(trunated_model, input_bound, method=method)
        trunated_model = self.truncate_model_at(tensor_name)
        input_names = [input_tensor.name for input_tensor in trunated_model.onnx.graph.input]
        return compute_model_bound(trunated_model._model, input_bound, input_names=input_names, method=method)
    
    def get_conv_wb(self, node):
        assert node.op_type == "Conv"
        _, w, b = node.input
        w = torch.from_numpy( numpy_helper.to_array(self._initializers_mapping[w]) )
        b = torch.from_numpy( numpy_helper.to_array(self._initializers_mapping[b]) )
        return w,b

    def get_gemm_wb(self, node):
        # Get the weights and bias for a GEMM node
        assert node.op_type == "Gemm"
        
        attr_dict = {attr.name: attr for attr in node.attribute}
        alpha = attr_dict['alpha'].f if 'alpha' in attr_dict else 1.0
        beta = attr_dict['beta'].f if 'beta' in attr_dict else 1.0
        transA = attr_dict['transA'].i if 'transA' in attr_dict else 0
        transB = attr_dict['transB'].i if 'transB' in attr_dict else 0

        _, w, b = node.input
        w = torch.from_numpy( numpy_helper.to_array(self._initializers_mapping[w]) )
        b = torch.from_numpy( numpy_helper.to_array(self._initializers_mapping[b]) )

        if transB == 0:
            w = w.t()

        return w,b

    @property
    def _bounded_model_gpu(self):
        if not hasattr(self, "_bounded_model_gpu_"):
            # self._bounded_model_gpu_ = BoundedModule(ConvertModel(self._model, experimental=True, quirks=custom_quirks), x, device='cuda')
            self._bounded_model_gpu_ = ConvertModel(self._model, experimental=True, quirks=custom_quirks).to('cuda')
        return self._bounded_model_gpu_
    @property
    def _bounded_model(self):
        if not hasattr(self, "_bounded_model_"):
            # self._bounded_model_ = BoundedModule(ConvertModel(self._model, experimental=True, quirks=custom_quirks), x)
            self._bounded_model_ = ConvertModel(self._model, experimental=True, quirks=custom_quirks).to('cpu')
        return self._bounded_model_
    
    def forward_gpu(self, x: torch.Tensor) -> torch.Tensor:
        # put tensor on gpu if not already
        if x.device != 'cuda':
            x = x.cuda()
        return self._bounded_model_gpu(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._bounded_model(x)

    def to(self, device):
        device = str(device).lower()
        if device == 'cuda':
            return self._bounded_model_gpu
        elif device == 'cpu':
            return self._bounded_model
        else:
            raise ValueError(f"Invalid device {device}")


    def forward_onnx(self, x: torch.Tensor) -> torch.Tensor:
        import onnxruntime as ort
        sess = InferenceSession(self._model.SerializeToString())
        return torch.tensor(sess.run(None, {'data': x.numpy()})[0])
        # return torch.tensor(sess.run(None, {'input': x.numpy()})[0])

    def save(self, fname: Path, downgrade=True):
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True)

        model = self._model
        if downgrade:
            model = version_converter.convert_version(model,11)
            model.ir_version = 7
        onnx.save(model, fname)
        return fname
    
    def generate_updated_model(self, nodes_to_replace, additional_nodes, additional_initializers, 
                      graph_name=f"bruh_split_merge",
                      producer_name="ReluSplitter"):
        # maybe some assertions
        new_nodes = []
        new_initializers = []
        model_in = self._model.graph.input
        model_out = self._model.graph.output
        model_input_name = model_in[0].name
        model_output_name = model_out[0].name

        visited = set([model_input_name])
        avaliable_nodes = [n for n in self._nodes if n not in nodes_to_replace] + additional_nodes
        avaliable_initializers = {i.name:i for i in (self._initializers+additional_initializers)}
        added_initializers = set()
        while model_output_name not in visited:
            updated = False
            for node in avaliable_nodes:
                if node in new_nodes:
                    continue
                else:
                    if all((i in visited or i in avaliable_initializers) for i in node.input):
                        new_nodes.append(node)

                        # new_initializers.extend([avaliable_initializers[i] for i in node.input if i in avaliable_initializers])
                        # initializer that are used multiple times produce invialid onnx model
                        initilizer_to_add = [i for i in node.input if i in avaliable_initializers and i not in added_initializers]
                        new_initializers.extend([avaliable_initializers[i] for i in initilizer_to_add])
                        added_initializers.update(initilizer_to_add)

                        visited.update(node.output)
                        updated = True
                        break
            if not updated:
                raise ValueError(f"Can't find a node to add to the new model")

        new_graph = onnx.helper.make_graph(
            new_nodes,
            graph_name,
            model_in,
            model_out,
            new_initializers
        )
        new_model = helper.make_model(new_graph, producer_name=producer_name, opset_imports=self.op_set)
        return WarppedOnnxModel(new_model, force_rename=False, logger=self.logger)

    def truncate_model_at(self, target_output_name,
                      graph_name=f"bruh_split_merge",
                      producer_name="ReluSplitter"):
        # maybe some assertions
        new_nodes = []
        new_initializers = []
        model_in = self._model.graph.input
        model_input_name = model_in[0].name

        visited = set([model_input_name])
        avaliable_nodes = self._nodes
        avaliable_initializers = {i.name:i for i in self._initializers}
        added_initializers = set()
        while target_output_name not in visited:
            updated = False
            for node in avaliable_nodes:
                if node in new_nodes:
                    continue
                else:
                    if all((i in visited or i in avaliable_initializers) for i in node.input):
                        new_nodes.append(node)
                        # new_initializers.extend([avaliable_initializers[i] for i in node.input if i in avaliable_initializers])
                        # initializer that are used multiple times produce invialid onnx model
                        initilizer_to_add = [i for i in node.input if i in avaliable_initializers and i not in added_initializers]
                        new_initializers.extend([avaliable_initializers[i] for i in initilizer_to_add])
                        added_initializers.update(initilizer_to_add)
                        visited.update(node.output)
                        updated = True
                        break
            if not updated:
                raise ValueError(f"Can't find a node to add to the new model")
        
        new_graph = onnx.helper.make_graph(
            new_nodes,
            graph_name,
            model_in,
            [onnx.helper.make_tensor_value_info(target_output_name, onnx.TensorProto.FLOAT, None)],
            new_initializers
        )
        new_model = helper.make_model(new_graph, producer_name=producer_name, opset_imports=self.op_set)
        return WarppedOnnxModel(new_model, force_rename=False, logger=self.logger)

    def replace_initializers(self, dicts):
        new_initializers = [i for i in self._initializers if i.name not in dicts.keys()]
        for k,v in dicts.items():
            # v is a troch tensor, need to compose it to onnx initializer
            new_initializers.append(onnx.helper.make_tensor(k, onnx.TensorProto.FLOAT, v.shape, v.flatten()))
        new_graph = onnx.helper.make_graph(
            self._nodes,
            self._model.graph.name,
            self._model.graph.input,
            self._model.graph.output,
            new_initializers
        )
        new_model = helper.make_model(new_graph, producer_name=self._model.producer_name, opset_imports=self.op_set)
        return WarppedOnnxModel(new_model, force_rename=False, logger=self.logger)
        
    
    @property
    def onnx(self):
        return self._model
    @property
    def ort_sess(self):
        return InferenceSession(self._model.SerializeToString())
        
    