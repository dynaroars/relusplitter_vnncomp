# standard libs
import os
import random
from functools import reduce

# third-party libs
import onnx
import torch
import numpy as np
from auto_LiRPA import BoundedTensor, PerturbationLpNorm

# my libs
from relu_splitter import TOOL_NAME
from relu_splitter.utils.read_vnnlib import read_vnnlib
from relu_splitter.utils.onnx_utils import check_models_closeness
from relu_splitter.model import WarppedOnnxModel
from relu_splitter.utils.logger import default_logger
from relu_splitter.anywhere.gemm import Rsplitter_Gemm
from relu_splitter.anywhere.conv import Rsplitter_Conv

# Ensure deterministic behavior (Mainly for Conv)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # or ":16:8" if needed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.enabled = False


class ReluSplitter_Anywhere(Rsplitter_Gemm, Rsplitter_Conv):
    supported_op_types = ["gemm", "conv"]
    supported_activations = ["relu", "relu", "prelu"]
    ignore_split_nodes = True
    supress_warnings = False
    logger = default_logger

    @classmethod
    def info(cls, network):
        model = WarppedOnnxModel.load(network)
        print("="*20 + " Model Info " + "="*20)
        print(f"Model: {network}")
        try:
            print(f"Model input shapes: {model.input_shapes}")
        except Exception as e:
            print(f"Error getting model info: {e}")

        # report splittable nodes
        splittable_nodes = cls.get_splittable_nodes_cls(model)
        print("-"*80)
        print(f"{'idx':10} | {'name':30} | {'op_type':10} | {'layer_size':20}")
        print("-"*80)
        for idx, node in enumerate(splittable_nodes):
            layer_size = None
            if node.op_type.lower() == "gemm":
                w,b = model.get_gemm_wb(node)
                layer_size = b.shape
            elif node.op_type.lower() == "conv":
                w,b = model.get_conv_wb(node)
                layer_size = b.shape
            is_split = TOOL_NAME in node.name
            # print a neat table
            print(f"{idx:10} | {node.name:30} | {node.op_type:10} | {str(layer_size):20}")

    @classmethod
    def get_splittable_nodes_cls(cls, warpped_model):
        splittable_nodes = []
        if not cls.supress_warnings and not cls.ignore_split_nodes:
            cls.logger.warning("ignore_split_nodes is set to False, splitting nodes that are already split might work suboptimally.")
        for node in warpped_model.nodes:
            if node.op_type.lower() in cls.supported_op_types:
                if cls.ignore_split_nodes and TOOL_NAME in node.name:
                    continue
                else:
                    splittable_nodes.append(node)

        return splittable_nodes

    def __init__(self, network, spec, input_shape=None, logger=default_logger):
        # gemm_node: the node to split (IT CAN BE ANY GEMM NODE IN THE MODEL)
        # model: the model containing the gemm_node
        self.onnx_path = network
        self.spec_path = spec
        self.logger = logger

        self.logger.debug("Initializing Rsplitter...")
        self.logger.debug(f"onnx: {self.onnx_path}, spec: {self.spec_path}, input_shape: {input_shape}")
        self.init_model()
        self.init_vnnlib()


    def init_vnnlib(self):
        input_bound, output_bound = read_vnnlib(str(self.spec_path))[0]
        input_lb, input_ub = torch.tensor([[[[i[0] for i in input_bound]]]]), torch.tensor([[[[i[1] for i in input_bound]]]])
        spec_num_inputs = reduce(lambda x, y: x*y, input_lb.shape)
        model_num_inputs = self.model.num_inputs
        input_lb, input_ub = input_lb.view(self.input_shape), input_ub.view(self.input_shape)               # reshape input bounds to match the model input shape
        # input_lb, input_ub = input_lb.reshape(1, *input_lb.shape), input_ub.reshape(1, *input_ub.shape)     # add batch dimension of 1  or lirpa will cry
        # add batch dimension of 1 if not already present
        if input_lb.shape[0] != 1:
            input_lb, input_ub = input_lb.reshape(1, *input_lb.shape), input_ub.reshape(1, *input_ub.shape)
        

        assert torch.all(input_lb <= input_ub), "Input lower bound is greater than upper bound"
        assert model_num_inputs == spec_num_inputs, f"Spec number of inputs does not match model inputs {spec_num_inputs} != {model_num_inputs}"

        self.bounded_input = BoundedTensor(input_lb, PerturbationLpNorm(norm=float("inf"), x_L=input_lb, x_U=input_ub))
        
        
    def init_model(self):
        self.model = WarppedOnnxModel.load(self.onnx_path)
        self.input_shape = list(self.model.input_shapes.values())[0]
        assert len(self.model.input) == 1, f"Model has more than one input {self.model.graph.input}"
        assert len(self.model.output) == 1, f"Model has more than one output {self.model.graph.output}"

    def init_seeds(self, random_seed):
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def get_splittable_nodes(self):
        return self.get_splittable_nodes_cls(self.model)
    
    def resolve_node_idx(self, op_type, idx):
        splitable_nodes = self.get_splittable_nodes()
        if op_type.lower() == "all":
            return splitable_nodes[idx]
        else:
            assert splitable_nodes[idx].op_type.lower() == op_type.lower(), f"Node at index {idx} is of type {splitable_nodes[idx].op_type}, expected {op_type}"
            return splitable_nodes[idx]
    
    def check_equivChk_conf(self, conf):
        return True

    def split(self, op_type, idx, conf):
        if "seed" in conf:
            self.init_seeds(conf["seed"])

        node = self.resolve_node_idx(op_type, idx)
        node_op_type = node.op_type
        self.logger.info(f"Splitting node {node.name} of type {node_op_type} at index {idx} with config: {conf}")
        if node_op_type == "Gemm":
            split_model, baseline_model = self.gemm_split(node, conf)
        elif node_op_type == "Conv":
            split_model, baseline_model = self.conv_split(node, conf)
        else:
            raise NotImplementedError(f"Splitting for op_type {node_op_type} is not implemented")
        
        if conf.get("equiv_chk_conf", None) is not None:
            equiv_chk_conf = conf["equiv_chk_conf"]
            closeness_results = check_models_closeness(
                self.model,
                [split_model, baseline_model],
                self.input_shape,
                device=None,
                n=equiv_chk_conf.get("n", 10),
                atol=equiv_chk_conf.get("atol", 1e-6),
                rtol=equiv_chk_conf.get("rtol", 1e-6)
            )
            (split_close, split_diff), (baseline_close, baseline_diff) = closeness_results
            
            print(f"Split model closeness: {split_close}, worst diff: {split_diff}")
            print(f"Baseline model closeness: {baseline_close}, worst diff: {baseline_diff}")

        return split_model, baseline_model

