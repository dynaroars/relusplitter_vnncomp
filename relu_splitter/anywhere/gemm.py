import random

import torch
import onnx
from tqdm import tqdm


from relu_splitter import TOOL_NAME

class Rsplitter_Gemm():

    def gemm_split(self, gemm_node, conf):
        split_activation = conf["split_activation"].lower()

        assert gemm_node.op_type == "Gemm", f"Node to split is not a Gemm node: {gemm_node.op_type}"
        assert split_activation in ["relu", "leakyrelu", "prelu"], f"Unsupported split activation: {split_activation}"

        split_model, baseline_model = self._gemm_split(gemm_node, conf)

        return split_model, baseline_model

    #----------------------- ReLU -----------------------#
    #----------------------- ReLU -----------------------#
    def _gemm_split(self, gemm_node, conf):
        # if create_baseline is True, create a baseline model, return will be (split_model, baseline_model)
        # if create_baseline is False, return will be the split_model ONLY
        bounding_method_tight = conf.get("bounding_method_tight", "backward")
        bounding_method_loose = conf.get("bounding_method_loose", "ibp")
        create_baseline = conf.get("create_baseline", False)

        split_activation = conf.get("split_activation", "relu").lower()
        additional_activation_conf = conf.get("additional_activation_conf", {})

        param_selection_conf = conf.get("param_conf", {})

        tight_bounds = self.model.get_bound_of(self.bounded_input, gemm_node.output[0], method=bounding_method_tight)
        loose_bounds = self.model.get_bound_of(self.bounded_input, gemm_node.output[0], method=bounding_method_loose)

        split_idxs = self._decide_split_idxs_gemm(tight_bounds, loose_bounds, conf)
        split_dict = self._decide_split_params_gemm(tight_bounds, loose_bounds, split_idxs, param_selection_conf)
        activation_params = {
            "leakyrelu_alpha": self._decide_leakyrelu_alpha_gemm(additional_activation_conf),
            "prelu_slope": self._decide_prelu_slope_gemm(additional_activation_conf),
        }
        new_model = self._split_general_gemm(gemm_node ,split_dict, split_activation, activation_params)

        baseline_model = None
        if create_baseline:
            baseline_idxs = []
            baseline_dict = self._decide_split_params_gemm(tight_bounds, loose_bounds, baseline_idxs, param_selection_conf)
            baseline_model = self._split_general_gemm(gemm_node ,baseline_dict, split_activation, activation_params)

        return new_model, baseline_model
    
    # responsible for selecting neurons to split
    def _decide_split_idxs_gemm(self, tight_bounds, loose_bounds, conf):
        sorting_strat = conf.get("candidiate_strat")   # random
        n_splits = conf.get("n_splits")
        seed = conf.get("seed")

        layer_width = tight_bounds[0].shape[1]
        self.logger.debug(f"Bound shape: lb {tight_bounds[0].shape}, ub {tight_bounds[1].shape}")
        idxs = list(range(layer_width))

        rnd = random.Random(seed)
        if n_splits == None:
            n_splits = layer_width
        assert 0 <= n_splits <= layer_width, f"n_splits {n_splits} must be between 0 and layer width {layer_width}, got {n_splits}"

        if sorting_strat == "random":
            rnd.shuffle(idxs)
        elif sorting_strat == "bound_width":
            # sort by bound width (ub - lb)
            idxs.sort(key=lambda i: (loose_bounds[1][0,i] - loose_bounds[0][0,i]).item(), reverse=True)
        else:
            raise NotImplementedError(f"sorting_strat {sorting_strat} is not implemented yet")
        
        split_idxs = idxs[:n_splits]
        self.logger.info(f"Splitting gemm neurons: {split_idxs}...")
        return split_idxs
            
    def _decide_split_params_gemm(self, tight_bounds, loose_bounds, split_idxs, split_conf):
        # if idx in split_idxs, use tight_bounds to decide tau (so split more likely to work)
        # if idx not in split_idxs, use loose_bounds to decide tau (stronger gaurantee of stability)
        gemm_tau_strat = split_conf.get("gemm_tau_strat").lower()
        split_scale_strat = split_conf.get("split_scale_strat").lower()
        if split_scale_strat == "fixed":
            fixed_scales = split_conf.get("fixed_scales")
        if split_scale_strat == "random":
            scale_range = split_conf.get("random_scale_range")
            s_pos_range = (scale_range[0], scale_range[1])
            s_neg_range = (-scale_range[1], -scale_range[0])

        stable_tau_strat = split_conf.get("stable_tau_strat").lower() # random, BigTau, SmallTau
        stable_tau_margin = split_conf.get("stable_tau_margin")
        cap_tau = split_conf.get("cap_tau")


        split_dict = {}
        layer_width = tight_bounds[0].shape[1]
        for i in range(layer_width):
            if i in split_idxs: # destabilized neuron
                lb, ub = tight_bounds[0][0,i].item(), tight_bounds[1][0,i].item()
                if gemm_tau_strat == "random":
                    tau = random.uniform(lb, ub)
                elif gemm_tau_strat == "midpoint":
                    tau = (lb+ub) / 2.0
                else:
                    raise NotImplementedError(f"gemm_tau_strat {gemm_tau_strat} is not implemented yet")
                # decide scales
                if split_scale_strat == "fixed":
                    s_pos, s_neg = fixed_scales
                elif split_scale_strat == "random":
                    s_pos = random.uniform(s_pos_range[0], s_pos_range[1])
                    s_neg = random.uniform(s_neg_range[0], s_neg_range[1])
                else:
                    raise NotImplementedError(f"split_scale_strat {split_scale_strat} is not implemented yet")
                
            else:   # Not destabilized neuron
                lb, ub = loose_bounds[0][0,i].item(), loose_bounds[1][0,i].item()
                _bigtau_tmp = abs(min(0, lb))
                bigtau = random.uniform(_bigtau_tmp + stable_tau_margin[0], _bigtau_tmp + stable_tau_margin[1])
                _smalltau_tmp = abs(max(0, ub))
                smalltau = -random.uniform(_smalltau_tmp + stable_tau_margin[0], _smalltau_tmp + stable_tau_margin[1])
                rand_tau = random.choice([bigtau, smalltau])
                if stable_tau_strat == "random":
                    tau = rand_tau
                elif stable_tau_strat == "big":
                    tau = bigtau
                elif stable_tau_strat == "small":
                    tau = smalltau
                else:
                    raise NotImplementedError(f"stable_tau_strat {stable_tau_strat} is not implemented yet")
                # decide scales
                if split_scale_strat == "fixed":
                    s_pos, s_neg = fixed_scales
                elif split_scale_strat == "random":
                    s_pos = random.uniform(s_pos_range[0], s_pos_range[1])
                    s_neg = random.uniform(s_neg_range[0], s_neg_range[1])
                else:
                    raise NotImplementedError(f"split_scale_strat {split_scale_strat} is not implemented yet")
            if abs(tau) > cap_tau:
                self.logger.warning(f"Tau value {tau} for neuron {i} exceeds cap {cap_tau}.")
                tau = max(min(tau, cap_tau), -cap_tau)
            self.logger.debug(f"Decided split params for neuron {i}: tau={tau}, s_pos={s_pos}, s_neg={s_neg}, destabilized={'Yes' if i in split_idxs else 'No'}")
            split_dict[i] = (tau, (s_pos, s_neg))
        return split_dict


    def _decide_leakyrelu_alpha_gemm(self, additional_activation_conf):
        return additional_activation_conf.get("leakyrelu_alpha")
    
    def _decide_prelu_slope_gemm(self, additional_activation_conf):
        range = additional_activation_conf.get("prelu_slope_range")
        return random.uniform(range[0], range[1])
        # return additional_activation_conf.get("prelu_slope", 0.25)


    # actually split
        
    def _split_general_gemm(self, node_to_split, split_dict, split_activation, params):
        if split_activation == "relu":
            split_model = self._split_ReLU_gemm(node_to_split, split_dict)
        elif split_activation == "leakyrelu":
            split_model = self._split_LeakyReLU_gemm(node_to_split, split_dict, alpha=params["leakyrelu_alpha"])
        elif split_activation == "prelu":
            split_model = self._split_PReLU_gemm(node_to_split, split_dict, slope=params["prelu_slope"])
        else:
            raise NotImplementedError(f"split_activation {split_activation} is not implemented yet")
        # b1 = self.model.get_bound_of(self.bounded_input, conv_node.output[0], method="backward")  # update bounds
        # b2 = split_model.get_bound_of(self.bounded_input, conv_node.output[0], method="backward")  # update bounds
        # print(f"Bound diff after conv split: lb diff {torch.abs(b1[0]-b2[0]).max().item()}, ub diff {torch.abs(b1[1]-b2[1]).max().item()}")
        return split_model

    def _split_ReLU_gemm(self, node_to_split,split_dict):
        # split_dict: {idx: (tau, (s_pos, s_neg))}

        w_o, b_o = self.model.get_gemm_wb(node_to_split)
        num_out, num_in = w_o.shape
        split_layer_width = 2 * num_out

        gemm_1_w = torch.zeros((split_layer_width, num_in))
        gemm_1_b = torch.zeros((split_layer_width,))
        gemm_2_w = torch.zeros((num_out, split_layer_width))
        gemm_2_b = torch.zeros(num_out)

        free_neurons = [i for i in range(split_layer_width)]
        random.shuffle(free_neurons)

        for i in tqdm(range(num_out), desc="Constructing new ReLU layers"):
            idx1, idx2 = free_neurons.pop(), free_neurons.pop()
            tau, (s_pos, s_neg) = split_dict[i]
            # gemm_1
            gemm_1_w[idx1]       = s_pos * w_o[i]
            gemm_1_w[idx2]   = s_neg * w_o[i]
            gemm_1_b[idx1]       = -s_pos * (tau - b_o[i])
            gemm_1_b[idx2]   = -s_neg * (tau - b_o[i])
            # gemm_2
            gemm_2_w[i][idx1]       = 1/s_pos
            gemm_2_w[i][idx2]   = 1/s_neg
            gemm_2_b[i]            = tau
            
        # gemm_1_w = gemm_1_w.T
        # gemm_2_w = gemm_2_w.T

        input_tname     = node_to_split.input[0]    # orginal input
        gemm_1_output_tname  = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm1")   # intermediate output after gemm_1
        relu_output_tname    = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_ReLU")    # intermediate output after relu
        gemm_2_output_tname  = node_to_split.output[0]                                   # final output after gemm_2, should be same as original output

        gemm1_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm1w")
        gemm1_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm1b")
        gemm2_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm2w")
        gemm2_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm2b")

        gemm1_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Gemm1")
        relu_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_ReLU")
        gemm2_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Gemm2")

        # create nodes
        gemm_1_node = onnx.helper.make_node(
            "Gemm",
            inputs=[input_tname, gemm1_w_tname, gemm1_b_tname],
            outputs=[gemm_1_output_tname],
            name=gemm1_nname,
            alpha=1.0,
            beta=1.0,
            transB=1,
            transA=0)
        relu_node = onnx.helper.make_node(
            "Relu",
            inputs=[gemm_1_output_tname],
            outputs=[relu_output_tname],
            name=relu_nname)
        gemm_2_node = onnx.helper.make_node(
            "Gemm",
            inputs=[relu_output_tname, gemm2_w_tname, gemm2_b_tname],
            outputs=[gemm_2_output_tname],
            name=gemm2_nname,
            alpha=1.0,
            beta=1.0,
            transB=1,
            transA=0)

        new_nodes = [gemm_1_node, relu_node, gemm_2_node]
        new_initializers = [
            onnx.helper.make_tensor(gemm1_w_tname, onnx.TensorProto.FLOAT, gemm_1_w.shape, gemm_1_w.flatten().tolist()),
            onnx.helper.make_tensor(gemm1_b_tname, onnx.TensorProto.FLOAT, gemm_1_b.shape, gemm_1_b.flatten().tolist()),
            onnx.helper.make_tensor(gemm2_w_tname, onnx.TensorProto.FLOAT, gemm_2_w.shape, gemm_2_w.flatten().tolist()),
            onnx.helper.make_tensor(gemm2_b_tname, onnx.TensorProto.FLOAT, gemm_2_b.shape, gemm_2_b.flatten().tolist()),
        ]

        new_model = self.model.generate_updated_model(
            nodes_to_replace=[node_to_split],
            additional_nodes=new_nodes,
            additional_initializers=new_initializers,
            graph_name=f"{self.model.graph_name}_GemmSplit",
            producer_name=TOOL_NAME
        )
        return new_model


    def _split_LeakyReLU_gemm(self, node_to_split,split_dict, alpha = 0.01):
        w_o, b_o = self.model.get_gemm_wb(node_to_split)
        num_out, num_in = w_o.shape
        split_layer_width = 2 * num_out

        gemm_1_w = torch.zeros((split_layer_width, num_in))
        gemm_1_b = torch.zeros((split_layer_width,))
        gemm_2_w = torch.zeros((num_out, split_layer_width))
        gemm_2_b = torch.zeros(num_out)

        alpha_scaling = 1.0 / (1.0 + alpha)

        free_neurons = [i for i in range(split_layer_width)]
        random.shuffle(free_neurons)

        for i in tqdm(range(num_out), desc="Constructing new LeakyReLU layers"):
            idx1, idx2 = free_neurons.pop(), free_neurons.pop()
            tau, (s_pos, s_neg) = split_dict[i]
            # gemm_1
            gemm_1_w[idx1]       = s_pos * w_o[i]
            gemm_1_w[idx2]   = s_neg * w_o[i]
            gemm_1_b[idx1]       = -s_pos * (tau - b_o[i])
            gemm_1_b[idx2]   = -s_neg * (tau - b_o[i])
            # gemm_2
            gemm_2_w[i][idx1]       = (1/s_pos) * alpha_scaling
            gemm_2_w[i][idx2]   = (1/s_neg) * alpha_scaling
            gemm_2_b[i]            = tau
            
        # gemm_1_w = gemm_1_w.T
        # gemm_2_w = gemm_2_w.T

        input_tname     = node_to_split.input[0]    # orginal input
        gemm_1_output_tname  = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm1")   # intermediate output after gemm_1
        leakyrelu_output_tname    = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_LeakyReLU")    # intermediate output after relu
        gemm_2_output_tname  = node_to_split.output[0]                                   # final output after gemm_2, should be same as original output
        gemm1_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm1w")
        gemm1_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm1b")
        gemm2_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm2w")
        gemm2_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm2b")
        gemm1_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Gemm1")
        leakyrelu_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_LeakyReLU")
        gemm2_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Gemm2")
        # create nodes
        gemm_1_node = onnx.helper.make_node(
            "Gemm",
            inputs=[input_tname, gemm1_w_tname, gemm1_b_tname],
            outputs=[gemm_1_output_tname],
            name=gemm1_nname,
            alpha=1.0,
            beta=1.0,
            transB=1,
            transA=0)
        leakyrelu_node = onnx.helper.make_node(
            "LeakyRelu",
            inputs=[gemm_1_output_tname],
            outputs=[leakyrelu_output_tname],
            name=leakyrelu_nname,
            alpha=alpha)
        gemm_2_node = onnx.helper.make_node(
            "Gemm",
            inputs=[leakyrelu_output_tname, gemm2_w_tname, gemm2_b_tname],
            outputs=[gemm_2_output_tname],
            name=gemm2_nname,
            alpha=1.0,
            beta=1.0,
            transB=1,
            transA=0)
        new_nodes = [gemm_1_node, leakyrelu_node, gemm_2_node]
        new_initializers = [
            onnx.helper.make_tensor(gemm1_w_tname, onnx.TensorProto.FLOAT, gemm_1_w.shape, gemm_1_w.flatten().tolist()),
            onnx.helper.make_tensor(gemm1_b_tname, onnx.TensorProto.FLOAT, gemm_1_b.shape, gemm_1_b.flatten().tolist()),
            onnx.helper.make_tensor(gemm2_w_tname, onnx.TensorProto.FLOAT, gemm_2_w.shape, gemm_2_w.flatten().tolist()),
            onnx.helper.make_tensor(gemm2_b_tname, onnx.TensorProto.FLOAT, gemm_2_b.shape, gemm_2_b.flatten().tolist()),
        ]
        new_model = self.model.generate_updated_model(
            nodes_to_replace=[node_to_split],
            additional_nodes=new_nodes,
            additional_initializers=new_initializers,
            graph_name=f"{self.model.graph_name}_GemmSplit_LeakyReLU",
            producer_name=TOOL_NAME
        )
        return new_model
    
    def _split_PReLU_gemm(self, node_to_split,split_dict, slope = 0.25):
        # slope is either a float or a list of floats (per-channel/per-neuron)
        w_o, b_o = self.model.get_gemm_wb(node_to_split)
        num_out, num_in = w_o.shape
        split_layer_width = 2 * num_out

        if isinstance(slope, float) or slope.ndim == 0:
            slope = torch.full((num_out,), slope)
        if isinstance(slope, torch.Tensor):
            assert slope.ndim == 1 and slope.shape[0] == num_out
        else:
            raise ValueError(f"Slope is not a valid float or 1D tensor: {slope}")
        
        
        gemm_1_w = torch.zeros((split_layer_width, num_in))
        gemm_1_b = torch.zeros((split_layer_width,))
        gemm_2_w = torch.zeros((num_out, split_layer_width))
        gemm_2_b = torch.zeros(num_out)
        actual_slope = torch.zeros((split_layer_width,))
        free_neurons = [i for i in range(split_layer_width)]
        random.shuffle(free_neurons)
        for i in tqdm(range(num_out), desc="Constructing new PReLU layers"):
            idx1, idx2 = free_neurons.pop(), free_neurons.pop()
            tau, (s_pos, s_neg) = split_dict[i]
            # gemm_1
            gemm_1_w[idx1]       = s_pos * w_o[i]
            gemm_1_w[idx2]   = s_neg * w_o[i]
            gemm_1_b[idx1]       = -s_pos * (tau - b_o[i])
            gemm_1_b[idx2]   = -s_neg * (tau - b_o[i])
            # prelu
            curr_slope = slope[i]
            actual_slope[idx1] = curr_slope
            actual_slope[idx2] = curr_slope
            # gemm_2
            slope_scaling = 1.0 / (1.0 + curr_slope)
            gemm_2_w[i][idx1]       = (1/s_pos) * slope_scaling
            gemm_2_w[i][idx2]   = (1/s_neg) * slope_scaling
            gemm_2_b[i]            = tau

        # gemm_1_w = gemm_1_w.T
        # gemm_2_w = gemm_2_w.T

        input_tname     = node_to_split.input[0]    # orginal input
        gemm_1_output_tname  = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm1")   # intermediate output after gemm_1
        prelu_output_tname    = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_PReLU")    # intermediate output after relu
        prelu_slope_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_PReLUslope")
        gemm_2_output_tname  = node_to_split.output[0]                                   # final output after gemm_2, should be same as original output
        gemm1_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm1w")
        gemm1_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm1b")
        gemm2_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm2w")
        gemm2_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Gemm2b")
        gemm1_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Gemm1")
        prelu_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_PReLU")
        gemm2_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Gemm2")
        # create nodes
        gemm_1_node = onnx.helper.make_node(
            "Gemm",
            inputs=[input_tname, gemm1_w_tname, gemm1_b_tname],
            outputs=[gemm_1_output_tname],
            name=gemm1_nname,
            alpha=1.0,
            beta=1.0,
            transB=1,
            transA=0)
        prelu_node = onnx.helper.make_node(
            "PRelu",
            inputs=[gemm_1_output_tname, prelu_slope_tname],
            outputs=[prelu_output_tname],
            name=prelu_nname)
        gemm_2_node = onnx.helper.make_node(
            "Gemm",
            inputs=[prelu_output_tname, gemm2_w_tname, gemm2_b_tname],
            outputs=[gemm_2_output_tname],
            name=gemm2_nname,
            alpha=1.0,
            beta=1.0,
            transB=1,
            transA=0)
        new_nodes = [gemm_1_node, prelu_node, gemm_2_node]
        new_initializers = [
            onnx.helper.make_tensor(gemm1_w_tname, onnx.TensorProto.FLOAT, gemm_1_w.shape, gemm_1_w.flatten().tolist()),
            onnx.helper.make_tensor(gemm1_b_tname, onnx.TensorProto.FLOAT, gemm_1_b.shape, gemm_1_b.flatten().tolist()),
            onnx.helper.make_tensor(gemm2_w_tname, onnx.TensorProto.FLOAT, gemm_2_w.shape, gemm_2_w.flatten().tolist()),
            onnx.helper.make_tensor(gemm2_b_tname, onnx.TensorProto.FLOAT, gemm_2_b.shape, gemm_2_b.flatten().tolist()),
            onnx.helper.make_tensor(prelu_slope_tname, onnx.TensorProto.FLOAT, actual_slope.shape, actual_slope.flatten().tolist()),
        ]
        new_model = self.model.generate_updated_model(
            nodes_to_replace=[node_to_split],
            additional_nodes=new_nodes,
            additional_initializers=new_initializers,
            graph_name=f"{self.model.graph_name}_GemmSplit_PReLU",
            producer_name=TOOL_NAME
        )
        return new_model
    