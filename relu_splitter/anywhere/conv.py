import random

import torch
import onnx
from tqdm import tqdm


from relu_splitter import TOOL_NAME


class Rsplitter_Conv():
    def conv_split(self, conv_node, conf):
        split_activation = conf["split_activation"].lower()

        assert conv_node.op_type == "Conv", f"Node to split is not a Conv node: {conv_node.op_type}"
        assert split_activation in ["relu", "leakyrelu", "prelu"], f"Unsupported split activation: {split_activation}"

        split_model, baseline_model = self._conv_split(conv_node, conf)

        return split_model, baseline_model
    

    #----------------------- Conv -----------------------#
    #----------------------- Conv -----------------------#
    def _conv_split(self, conv_node, conf):
        bounding_method_tight = conf.get("bounding_method_tight", "backward")
        bounding_method_loose = conf.get("bounding_method_loose", "ibp")
        create_baseline = conf.get("create_baseline", False)

        split_activation = conf.get("split_activation", "relu").lower()
        additional_activation_params = conf.get("additional_activation_params", {})

        param_selection_conf = conf.get("param_conf", {})

        # Tight bounds shape: torch.Size([1, 8, 16, 16])
        tight_bounds = self.model.get_bound_of(self.bounded_input, conv_node.output[0], method=bounding_method_tight)
        loose_bounds = self.model.get_bound_of(self.bounded_input, conv_node.output[0], method=bounding_method_loose)
        tight_bounds = (tight_bounds[0].squeeze(0), tight_bounds[1].squeeze(0))  # remove batch dim
        loose_bounds = (loose_bounds[0].squeeze(0), loose_bounds[1].squeeze(0))  # remove batch dim

        split_idxs = self._decide_split_idxs_conv(tight_bounds, loose_bounds, conf)
        split_dict = self._decide_split_params_conv(tight_bounds, loose_bounds, split_idxs, param_selection_conf)
        activation_params = {
            "leakyrelu_alpha": self._decide_leakyrelu_alpha_conv(additional_activation_params),
            "prelu_slope": self._decide_prelu_slope_conv(additional_activation_params),
        }
        new_model = self._split_general_conv(conv_node ,split_dict, split_activation, activation_params)

        baseline_model = None
        if create_baseline:
            baseline_idxs = []
            baseline_dict = self._decide_split_params_conv(tight_bounds, loose_bounds, baseline_idxs, param_selection_conf)
            baseline_model = self._split_general_conv(conv_node ,baseline_dict, split_activation, activation_params)

        return new_model, baseline_model
    
    def _decide_split_idxs_conv(self, tight_bounds, loose_bounds, conf):
        sorting_strat = conf.get("candidiate_strat")   # random
        n_splits = conf.get("n_splits")
        seed = conf.get("seed")

        layer_width = tight_bounds[0].shape[0]  # number of kernels
        idxs = list(range(layer_width))

        rnd = random.Random(seed)
        if n_splits == None:
            n_splits = layer_width
        assert 0 <= n_splits <= layer_width, f"n_splits {n_splits} must be between 0 and layer width {layer_width}, got {n_splits}"

        if sorting_strat == "random":
            rnd.shuffle(idxs)
        else:
            raise NotImplementedError(f"sorting_strat {sorting_strat} is not implemented for CONV yet")
        
        split_idxs = idxs[:n_splits]
        self.logger.info(f"Splitting conv filters: {split_idxs}...")
        return split_idxs

    def _decide_split_params_conv(self, tight_bounds, loose_bounds, split_idxs, param_selection_conf):
        conv_tau_strat = param_selection_conf.get("conv_tau_strat", "naive_sliding_window")
        conv_scale_strat = param_selection_conf.get("split_scale_strat")
        if conv_scale_strat == "fixed":
            fixed_scale = param_selection_conf.get("fixed_scales")
        if conv_scale_strat == "random":
            scale_range = param_selection_conf.get("random_scale_range")
            s_pos_range = (scale_range[0], scale_range[1])
            s_neg_range = (-scale_range[1], -scale_range[0])

        stable_tau_strat = param_selection_conf.get("stable_tau_strat").lower() # random, BigTau, SmallTau
        stable_tau_margin = param_selection_conf.get("stable_tau_margin")
        cap_tau = param_selection_conf.get("cap_tau")

        n_kernels = tight_bounds[0].shape[0]

        split_dict = {}
        for idx in range(n_kernels):
            tight_kernel_lb, tight_kernel_ub = tight_bounds[0][idx], tight_bounds[1][idx]
            loose_kernel_lb, loose_kernel_ub = loose_bounds[0][idx], loose_bounds[1][idx]
            if idx in split_idxs:   # require split
                # taus
                if conv_tau_strat == "naive_sliding_window":
                    # find the val to sat most intervals using sliding line sweep
                    evnts = []
                    for i in range(tight_kernel_lb.numel()):
                        evnts.append( (tight_kernel_lb.view(-1)[i].item(), 1) )
                        evnts.append( (tight_kernel_ub.view(-1)[i].item(), -1))
                    evnts.sort(key=lambda x: x[0])
                    active_intervals = 0
                    max_active_intervals = -1
                    start, end = None, None
                    curr_start = evnts[0][0]
                    for val, evnt in evnts:
                        active_intervals += evnt
                        if active_intervals > max_active_intervals:
                            max_active_intervals = active_intervals
                            start = curr_start
                            end = val
                        curr_start = val
                    tau = random.uniform(start, end)
                    self.logger.debug(f"Kernel {idx} naive sliding window tau range: ({start}, {end}), num_intervals: {max_active_intervals}/{tight_kernel_lb.numel()}")
                else:
                    raise NotImplementedError(f"conv_tau_strat {conv_tau_strat} is not implemented yet")
                # scales
                if conv_scale_strat == "fixed":
                    s_pos, s_neg = fixed_scale
                elif conv_scale_strat == "random":
                    s_pos = random.uniform(s_pos_range[0], s_pos_range[1])
                    s_neg = random.uniform(s_neg_range[0], s_neg_range[1])
                else:
                    raise NotImplementedError(f"conv_scale_strat {conv_scale_strat} is not implemented yet")
                self.logger.debug(f"Kernel {idx} split, tau: {tau}, s_pos: {s_pos}, s_neg: {s_neg}")
            else:   # no split
                # decide stable tau
                _bigtau_tmp = abs(min(loose_kernel_lb.min().item(), 0.0))
                bigtau = random.uniform(_bigtau_tmp + stable_tau_margin[0], _bigtau_tmp + stable_tau_margin[1])
                _smalltau_tmp = abs(max(loose_kernel_ub.max().item(), 0.0))
                smalltau = -random.uniform(stable_tau_margin[0], stable_tau_margin[1]) - _smalltau_tmp
                random_tau = random.choice([bigtau, smalltau])
                if stable_tau_strat == "big":
                    tau = bigtau
                elif stable_tau_strat == "small":
                    tau = smalltau
                elif stable_tau_strat == "random":
                    tau = random_tau
                else:
                    raise NotImplementedError(f"stable_tau_strat {stable_tau_strat} is not implemented yet")
                # decide stable scales
                if conv_scale_strat == "fixed":
                    s_pos, s_neg = fixed_scale
                elif conv_scale_strat == "random":
                    s_pos = random.uniform(s_pos_range[0], s_pos_range[1])
                    s_neg = random.uniform(s_neg_range[0], s_neg_range[1])
                else:
                    raise NotImplementedError(f"conv_scale_strat {conv_scale_strat} is not implemented yet")
            if abs(tau) > cap_tau:
                self.logger.warning(f"Tau value {tau} for kernel {idx} exceeds cap {cap_tau}.")
                tau = max(min(tau, cap_tau), -cap_tau)
            self.logger.debug(f"Decided split params for kernel {idx}: tau={tau}, s_pos={s_pos}, s_neg={s_neg}, destabilized={'Yes' if idx in split_idxs else 'No'} ")
            split_dict[idx] = (tau, (s_pos, s_neg))
        return split_dict

    def _decide_leakyrelu_alpha_conv(self, additional_activation_conf):
        return additional_activation_conf.get("leakyrelu_alpha", 0.01)
    
    def _decide_prelu_slope_conv(self, additional_activation_conf):
        return additional_activation_conf.get("prelu_slope", 0.25)
    
    def _split_general_conv(self, conv_node, split_dict, split_activation, params):
        if split_activation == "relu":
            split_model = self._split_ReLU_conv(conv_node, split_dict)
        elif split_activation == "leakyrelu":
            split_model = self._split_LeakyReLU_conv(conv_node, split_dict, params["leakyrelu_alpha"])
        elif split_activation == "prelu":
            split_model = self._split_PReLU_conv(conv_node, split_dict, params["prelu_slope"])
        else:
            raise NotImplementedError(f"split_activation {split_activation} is not implemented yet")
        # b1 = self.model.get_bound_of(self.bounded_input, conv_node.output[0], method="backward")  # update bounds
        # b2 = split_model.get_bound_of(self.bounded_input, conv_node.output[0], method="backward")  # update bounds
        # print(f"Bound diff after conv split: lb diff {torch.abs(b1[0]-b2[0]).max().item()}, ub diff {torch.abs(b1[1]-b2[1]).max().item()}")
        return split_model
    
    def _split_ReLU_conv(self, node_to_split, split_dict):

        ori_dilations = node_to_split.attribute[1].ints
        if ori_dilations == []:
            ori_dilations = [1,1]
        ori_kernel_shape = node_to_split.attribute[2].ints
        ori_pads = node_to_split.attribute[3].ints
        ori_strides = node_to_split.attribute[4].ints

        ori_w, ori_b = self.model.get_conv_wb(node_to_split)
        assert ori_w.dim() == 4, f"Conv weight dimension is not 4: {ori_w.dim()}"
        ori_oC, ori_iC, ori_kH, ori_kW = ori_w.shape

        split_layer_kernel_count = 2 * ori_oC

        conv1_w = torch.zeros((split_layer_kernel_count, ori_iC, ori_kH, ori_kW))
        conv1_b = torch.zeros((split_layer_kernel_count,))
        conv2_w = torch.zeros((ori_oC, split_layer_kernel_count, 1, 1))
        conv2_b = torch.zeros((ori_oC,))

        free_kernels = [i for i in range(split_layer_kernel_count)]
        random.shuffle(free_kernels)

        for i in tqdm(range(ori_oC), desc="Creating split Conv with ReLU"):
            tau, (s_pos, s_neg) = split_dict[i]
            # first conv layer
            idx1 = free_kernels.pop()
            idx2 = free_kernels.pop()
            
            conv1_w[idx1] =  s_pos * ori_w[i]
            conv1_w[idx2] =  s_neg * ori_w[i]
            conv1_b[idx1] =  -s_pos * (tau - ori_b[i])
            conv1_b[idx2] =  -s_neg * (tau - ori_b[i])
            # # set merge w & b
            conv2_w[i, idx1, 0, 0] = 1.0/s_pos
            conv2_w[i, idx2, 0, 0] = 1.0/s_neg
            conv2_b[i] = tau
        
        conv1_input_tname = node_to_split.input[0]
        conv1_output_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv1")
        relu_output_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_ReLU")
        conv2_output_tname = node_to_split.output[0]

        conv1_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv1_w")
        conv1_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv1_b")
        conv2_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv2_w")
        conv2_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv2_b")

        conv1_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Conv1")
        relu_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_ReLU")
        conv2_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Conv2")

        conv1_node = onnx.helper.make_node(
            'Conv',
            inputs=[conv1_input_tname, conv1_w_tname, conv1_b_tname],
            outputs=[conv1_output_tname],
            name=conv1_nname,
            dilations=ori_dilations,
            kernel_shape=ori_kernel_shape,
            pads=ori_pads,
            strides=ori_strides
        )
        relu_node = onnx.helper.make_node(
            'Relu',
            inputs=[conv1_output_tname],
            outputs=[relu_output_tname],
            name=relu_nname
        )
        conv2_node = onnx.helper.make_node(
            'Conv',
            inputs=[relu_output_tname, conv2_w_tname, conv2_b_tname],
            outputs=[conv2_output_tname],
            name=conv2_nname,
            dilations=[1,1],
            kernel_shape=[1,1],
            pads=[0,0,0,0],
            strides=[1,1]
        )
        
        new_nodes = [conv1_node, relu_node, conv2_node]
        new_initializers = [
            onnx.helper.make_tensor(conv1_w_tname, onnx.TensorProto.FLOAT, conv1_w.shape, conv1_w.flatten().tolist()),
            onnx.helper.make_tensor(conv1_b_tname, onnx.TensorProto.FLOAT, conv1_b.shape, conv1_b.flatten().tolist()),
            onnx.helper.make_tensor(conv2_w_tname, onnx.TensorProto.FLOAT, conv2_w.shape, conv2_w.flatten().tolist()),
            onnx.helper.make_tensor(conv2_b_tname, onnx.TensorProto.FLOAT, conv2_b.shape, conv2_b.flatten().tolist()),
        ]

        new_model = self.model.generate_updated_model(
            nodes_to_replace=[node_to_split],
            additional_nodes=new_nodes,
            additional_initializers=new_initializers,
            graph_name=f"{self.model.graph_name}_ConvSplit_ReLU",
            producer_name=TOOL_NAME
        )

        return new_model


    def _split_LeakyReLU_conv(self, node_to_split, split_dict, alpha):
        
        ori_dilations = node_to_split.attribute[1].ints
        if ori_dilations == []:
            ori_dilations = [1,1]
        ori_kernel_shape = node_to_split.attribute[2].ints
        ori_pads = node_to_split.attribute[3].ints
        ori_strides = node_to_split.attribute[4].ints

        ori_w, ori_b = self.model.get_conv_wb(node_to_split)
        assert ori_w.dim() == 4, f"Conv weight dimension is not 4: {ori_w.dim()}"
        ori_oC, ori_iC, ori_kH, ori_kW = ori_w.shape

        split_layer_kernel_count = 2 * ori_oC

        conv1_w = torch.zeros((split_layer_kernel_count, ori_iC, ori_kH, ori_kW))
        conv1_b = torch.zeros((split_layer_kernel_count,))
        conv2_w = torch.zeros((ori_oC, split_layer_kernel_count, 1, 1))
        conv2_b = torch.zeros((ori_oC,))

        free_kernels = [i for i in range(split_layer_kernel_count)]
        random.shuffle(free_kernels)

        alpha_scaling = 1.0 / (1.0 + alpha)

        for i in tqdm(range(ori_oC), desc="Creating split Conv with LeakyReLU"):
            tau, (s_pos, s_neg) = split_dict[i]
            # first conv layer
            idx1, idx2 = free_kernels.pop(), free_kernels.pop()
            
            conv1_w[idx1] =  s_pos * ori_w[i]
            conv1_w[idx2] =  s_neg * ori_w[i]
            conv1_b[idx1] =  -s_pos * (tau - ori_b[i])
            conv1_b[idx2] =  -s_neg * (tau - ori_b[i])
            # # set merge w & b
            conv2_w[i, idx1, 0, 0] = (1.0/s_pos) * alpha_scaling
            conv2_w[i, idx2, 0, 0] = (1.0/s_neg) * alpha_scaling
            conv2_b[i] = tau
        
        conv1_input_tname = node_to_split.input[0]
        conv1_output_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv1")
        leakyrelu_output_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_LeakyReLU")
        conv2_output_tname = node_to_split.output[0]

        conv1_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv1_w")
        conv1_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv1_b")
        conv2_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv2_w")
        conv2_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv2_b")

        conv1_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Conv1")
        leakyrelu_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_LeakyReLU")
        conv2_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Conv2")

        conv1_node = onnx.helper.make_node(
            'Conv',
            inputs=[conv1_input_tname, conv1_w_tname, conv1_b_tname],
            outputs=[conv1_output_tname],
            name=conv1_nname,
            dilations=ori_dilations,
            kernel_shape=ori_kernel_shape,
            pads=ori_pads,
            strides=ori_strides
        )
        leakyrelu_node = onnx.helper.make_node(
            'LeakyRelu',
            inputs=[conv1_output_tname],
            outputs=[leakyrelu_output_tname],
            name=leakyrelu_nname,
            alpha=alpha
        )
        conv2_node = onnx.helper.make_node(
            'Conv',
            inputs=[leakyrelu_output_tname, conv2_w_tname, conv2_b_tname],
            outputs=[conv2_output_tname],
            name=conv2_nname,
            dilations=[1,1],
            kernel_shape=[1,1],
            pads=[0,0,0,0],
            strides=[1,1]
        )

        new_nodes = [conv1_node, leakyrelu_node, conv2_node]
        new_initializers = [
            onnx.helper.make_tensor(conv1_w_tname, onnx.TensorProto.FLOAT, conv1_w.shape, conv1_w.flatten().tolist()),
            onnx.helper.make_tensor(conv1_b_tname, onnx.TensorProto.FLOAT, conv1_b.shape, conv1_b.flatten().tolist()),
            onnx.helper.make_tensor(conv2_w_tname, onnx.TensorProto.FLOAT, conv2_w.shape, conv2_w.flatten().tolist()),
            onnx.helper.make_tensor(conv2_b_tname, onnx.TensorProto.FLOAT, conv2_b.shape, conv2_b.flatten().tolist()),
        ]

        new_model = self.model.generate_updated_model(
            nodes_to_replace=[node_to_split],
            additional_nodes=new_nodes,
            additional_initializers=new_initializers,
            graph_name=f"{self.model.graph_name}_ConvSplit_LeakyReLU",
            producer_name=TOOL_NAME
        )
        return new_model
    
    def _split_PReLU_conv(self, node_to_split, split_dict, slope):
        ori_dilations = node_to_split.attribute[1].ints
        if ori_dilations == []:
            ori_dilations = [1,1]
        ori_kernel_shape = node_to_split.attribute[2].ints
        ori_pads = node_to_split.attribute[3].ints
        ori_strides = node_to_split.attribute[4].ints

        ori_w, ori_b = self.model.get_conv_wb(node_to_split)
        assert ori_w.dim() == 4, f"Conv weight dimension is not 4: {ori_w.dim()}"
        ori_oC, ori_iC, ori_kH, ori_kW = ori_w.shape

        split_layer_kernel_count = 2 * ori_oC

        if isinstance(slope, float) or slope.ndim == 0:
            slope = torch.full((ori_oC,), slope)
        if isinstance(slope, torch.Tensor):
            assert slope.ndim == 1 and slope.shape[0] == ori_oC
        else:
            raise ValueError(f"Slope is not a valid float or 1D tensor: {slope}")

        conv1_w = torch.zeros((split_layer_kernel_count, ori_iC, ori_kH, ori_kW))
        conv1_b = torch.zeros((split_layer_kernel_count,))
        conv2_w = torch.zeros((ori_oC, split_layer_kernel_count, 1, 1))
        conv2_b = torch.zeros((ori_oC,))
        actual_slope = torch.zeros((split_layer_kernel_count,1,1))

        free_kernels = [i for i in range(split_layer_kernel_count)]
        random.shuffle(free_kernels)


        for i in tqdm(range(ori_oC), desc="Creating split Conv with LeakyReLU"):
            tau, (s_pos, s_neg) = split_dict[i]
            # first conv layer
            idx1, idx2 = free_kernels.pop(), free_kernels.pop()
            
            conv1_w[idx1] =  s_pos * ori_w[i]
            conv1_w[idx2] =  s_neg * ori_w[i]
            conv1_b[idx1] =  -s_pos * (tau - ori_b[i])
            conv1_b[idx2] =  -s_neg * (tau - ori_b[i])
            actual_slope[idx1,0,0] = slope[i]
            actual_slope[idx2,0,0] = slope[i]
            # # set merge w & b
            slope_scaling = 1.0 / (1.0 + slope[i])
            conv2_w[i, idx1, 0, 0] = (1.0/s_pos) * slope_scaling
            conv2_w[i, idx2, 0, 0] = (1.0/s_neg) * slope_scaling
            conv2_b[i] = tau
        
        conv1_input_tname = node_to_split.input[0]
        conv1_output_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv1")
        leakyrelu_output_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_LeakyReLU")
        conv2_output_tname = node_to_split.output[0]

        conv1_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv1_w")
        conv1_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv1_b")
        conv2_w_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv2_w")
        conv2_b_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_Conv2_b")
        prelu_slope_tname = self.model.gen_tensor_name(prefix=f"{TOOL_NAME}_PReLU_slope")

        conv1_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Conv1")
        leakyrelu_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_LeakyReLU")
        conv2_nname = self.model.gen_node_name(prefix=f"{TOOL_NAME}_Conv2")

        conv1_node = onnx.helper.make_node(
            'Conv',
            inputs=[conv1_input_tname, conv1_w_tname, conv1_b_tname],
            outputs=[conv1_output_tname],
            name=conv1_nname,
            dilations=ori_dilations,
            kernel_shape=ori_kernel_shape,
            pads=ori_pads,
            strides=ori_strides
        )
        prelu_node = onnx.helper.make_node(
            'PRelu',
            inputs=[conv1_output_tname, prelu_slope_tname],
            outputs=[leakyrelu_output_tname],
            name=leakyrelu_nname,
        )
        conv2_node = onnx.helper.make_node(
            'Conv',
            inputs=[leakyrelu_output_tname, conv2_w_tname, conv2_b_tname],
            outputs=[conv2_output_tname],
            name=conv2_nname,
            dilations=[1,1],
            kernel_shape=[1,1],
            pads=[0,0,0,0],
            strides=[1,1]
        )

        new_nodes = [conv1_node, prelu_node, conv2_node]
        new_initializers = [
            onnx.helper.make_tensor(conv1_w_tname, onnx.TensorProto.FLOAT, conv1_w.shape, conv1_w.flatten().tolist()),
            onnx.helper.make_tensor(conv1_b_tname, onnx.TensorProto.FLOAT, conv1_b.shape, conv1_b.flatten().tolist()),
            onnx.helper.make_tensor(conv2_w_tname, onnx.TensorProto.FLOAT, conv2_w.shape, conv2_w.flatten().tolist()),
            onnx.helper.make_tensor(conv2_b_tname, onnx.TensorProto.FLOAT, conv2_b.shape, conv2_b.flatten().tolist()),
            onnx.helper.make_tensor(prelu_slope_tname, onnx.TensorProto.FLOAT, actual_slope.shape, actual_slope.flatten().tolist()),
        ]

        new_model = self.model.generate_updated_model(
            nodes_to_replace=[node_to_split],
            additional_nodes=new_nodes,
            additional_initializers=new_initializers,
            graph_name=f"{self.model.graph_name}_ConvSplit_LeakyReLU",
            producer_name=TOOL_NAME
        )
        return new_model