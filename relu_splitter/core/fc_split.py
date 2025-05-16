from .common import *

class RSplitter_fc():
    # def get_split_loc_fn_fc(self, nodes, **kwargs):
    #     gemm_node, relu_node = nodes
    #     fc_strategy = self._conf["fc_strategy"]
    #     lb, ub = self.warpped_model.get_bound_of(self.bounded_input, gemm_node.input[0])          # the input bound of the Gemm node, from which the split location is sampled
    #     lb, ub = lb.squeeze(), ub.squeeze()
    #     if fc_strategy == "single":
    #         split_loc = (torch.rand_like(lb) * (ub - lb) + lb)
    #         return lambda **kwargs: split_loc
    #     elif fc_strategy == "random":
    #         return lambda **kwargs: (torch.rand_like(lb) * (ub - lb) + lb)
    #     elif fc_strategy == "reluS+":
    #         def split_loc_fn(**kwargs):
    #             w, b = kwargs["w"], kwargs["b"]
    #             return find_feasible_point(lb, ub, w, b)
    #         return split_loc_fn
    #     elif fc_strategy == "reluS-":
    #         def split_loc_fn(**kwargs):
    #             w, b = kwargs["w"], kwargs["b"]
    #             return find_feasible_point(lb, ub, -w, -b)
    #         return split_loc_fn
    #     else:
    #         raise INVALID_PARAMETER(f"Unknown split strategy {fc_strategy}")

    def fc_info(self, idx, mode, bounding_method):
        gemm_node, relu_node = self.resolve_idx(idx, mode)
        bounds = self.warpped_model.get_bound_of(self.bounded_input, gemm_node.output[0], method=bounding_method)
        masks = self.fc_get_split_masks(bounds)
        info = {k: torch.sum(v).item() for k,v in masks.items()}
        return info


    def fc_get_split_masks(self, bounds):
        output_lb, output_ub = bounds
        masks = {}
        masks["all"] = torch.ones_like(output_lb, dtype=torch.bool).squeeze()
        masks["stable"] = torch.logical_or(output_lb > 0, output_ub <= 0).squeeze()
        masks["unstable"] = ~masks["stable"]
        masks["stable+"] = (output_lb > 0).squeeze()
        masks["stable-"] = (output_ub <= 0).squeeze()
        masks["unstable_n_stable+"] = torch.logical_or(masks["unstable"], masks["stable+"])
        assert torch.all(masks["stable"] == (masks["stable+"] | masks["stable-"])), "stable is not equal to stable+ AND stable-"
        assert not torch.any(masks["stable+"] & masks["stable-"]), "stable+ and stable- are not mutually exclusive"
        assert not torch.any(masks["unstable"] & masks["stable"]), "unstable and stable are not mutually exclusive"
        assert torch.all((masks["unstable"] | masks["stable"]) == masks["all"]), "The union of unstable and stable does not cover all elements"
        return masks


    def split_fc(self, nodes_to_split, n_splits=None, split_mask="stable", fc_strategy="random", scale_factors=(1.0, -1.0), create_baseline=False, bounding_method="backward"):
        gemm_node, relu_node = nodes_to_split
        assert all(attri.i == 0 for attri in gemm_node.attribute if attri.name == "TransA"), "TransA == 1 is not supported yet"

        self.logger.debug("=====================================")
        self.logger.debug(f"Splitting model: {self.onnx_path} with spec: {self.spec_path}")
        self.logger.debug(f"Splitting at Gemm node: <{gemm_node.name}> && ReLU node: <{relu_node.name}>")
        # self.logger.debug(f"Split strategy: {fc_strategy}")
        self.logger.debug(f"Split mask: {split_mask}")

        bounds = self.warpped_model.get_bound_of(self.bounded_input, gemm_node.output[0], method=bounding_method)
        split_masks = self.fc_get_split_masks(bounds)
        
        self.logger.info(f"============= Split Mask Sumamry =============")
        self.logger.info(f"stable+: {torch.sum(split_masks['stable+'])}\t"
                            f"stable-: {torch.sum(split_masks['stable-'])}")
        self.logger.info(f"unstable: {torch.sum(split_masks['unstable'])}\t"
                            f"all: {torch.sum(split_masks['all'])}")

        _split_mask = split_masks[split_mask]
        mask_size  = torch.sum(_split_mask).item()

        if n_splits is None:
            n_splits = mask_size
        if mask_size == 0 or mask_size < n_splits:
            # randomly pick 10 ~ 35 % of the neurons to split (0.1~0.35) * masks["all"]
            # really bad code: if no splitable neuron we just randomly pick 10-35% of the neurons to split
            _split_mask = split_masks["all"]
            n_splits = int(random.uniform(0.1, 0.35) * torch.sum(split_masks['all']))
            _split_mask = adjust_mask_random_k(_split_mask, n_splits)

            # raise INVALID_PARAMETER(f"Not enough {split_mask} neuron to split... (Requested: {n_splits}, Available: {mask_size})")
        if n_splits <= mask_size:
            _split_mask = adjust_mask_first_k(_split_mask, n_splits)
            
        
        _n_splits = torch.sum(_split_mask).item()  # actual number of splits
        split_idxs = torch.nonzero(_split_mask).squeeze().tolist()   # idx of non-zero elements in the split mask
        if isinstance(split_idxs, int):         # the line above returns a single int if there is only one split
            split_idxs = [split_idxs]
        self.logger.info(f"Selecting {_n_splits}/{mask_size} {split_mask} ReLUs to split")

        # TODO: put the split loc implementation here
        if fc_strategy == "random":
            split_offsets = [random.uniform(bounds[0][0,i], bounds[1][0,i]) for i in split_idxs]
        # elif fc_strategy == "reluS+":
        #     split_offsets = [random.uniform(max(0,bounds[0][0,i]), bounds[1][0,i]) for i in split_idxs]
        # elif fc_strategy == "reluS-":
            # split_offsets = [random.uniform(bounds[0][0,i], min(0,bounds[1][0,i])) for i in split_idxs]
        else:
            raise INVALID_PARAMETER(f"Unknown split strategy {fc_strategy}")
        assert len(split_offsets) == len(split_idxs) == _n_splits

        # Create the models
        split_model = self._split_fc(nodes_to_split, split_idxs, split_offsets, scale_factors)
        baseline_model = self._split_fc_baseline(nodes_to_split, split_idxs) if create_baseline else None

        return (split_model, baseline_model)


    def _split_fc(self, nodes_to_split, split_idxs=[], split_offsets=[], scale_factors=(1.0, -1.0)):
        gemm_node, relu_node = nodes_to_split
        n_splits = len(split_idxs)

        w_o, b_o = self.warpped_model.get_gemm_wb(gemm_node)    # w,b of the layer to be splitted
        # create new layers
        num_out, num_in = w_o.shape
        split_weights = np.zeros((num_out + n_splits, num_in))
        split_bias = np.zeros(num_out + n_splits)
        merge_weights = np.zeros((num_out, num_out + n_splits))
        merge_bias = np.zeros(num_out)
        
        idx = 0     # index of neuron in the new split layer
        for i in tqdm(range(num_out), desc="Constructing new layers"):
            if i in split_idxs:
                split_offset = split_offsets[split_idxs.index(i)]
                scale_ratio_pos, scale_ratio_neg = scale_factors

                split_weights[idx]      = scale_ratio_pos * w_o[i]
                split_weights[idx+1]    = scale_ratio_neg * w_o[i]
                split_bias[idx]         = -scale_ratio_pos * (split_offset - b_o[i])
                split_bias[idx+1]       = -scale_ratio_neg * (split_offset - b_o[i])

                merge_weights[i][idx]   = 1/scale_ratio_pos
                merge_weights[i][idx+1] = 1/scale_ratio_neg
                merge_bias[i] = split_offset

                idx += 2
            else:
                split_weights[idx] = w_o[i]
                split_bias[idx] = b_o[i]
                merge_weights[i][idx] = 1
                idx += 1
        split_weights = split_weights.T
        merge_weights = merge_weights.T
        
        orginal_input_name = gemm_node.input[0]
        orginal_output_name = relu_node.output[0]
        split_pre_tensor_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_sPre")
        split_post_tensor_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_sPost")
        merge_pre_tensor_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_mPre")
        merge_pose_tensor_name = orginal_output_name
        split_w_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_sW")
        split_b_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_sB")
        merge_w_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_mW")
        merge_b_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_mB")
        split_node_name      = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}_s")
        split_relu_node_name = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}_sR")
        merge_node_name      = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}_m")
        merge_relu_node_name = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}_mR")

        # create the new split layer
        split_layer = onnx.helper.make_node("Gemm",
                                            inputs=[orginal_input_name, split_w_name, split_b_name],
                                            outputs=[split_pre_tensor_name],
                                            name=split_node_name,
                                            alpha=1.0,
                                            beta=1.0,
                                            transB=0,
                                            transA=0)
        split_layer_relu = onnx.helper.make_node("Relu",
                                                inputs=[split_pre_tensor_name],
                                                outputs=[split_post_tensor_name],
                                                name=split_relu_node_name)
        # create the new merge layer
        merge_layer = onnx.helper.make_node("Gemm",
                                            inputs=[split_post_tensor_name, merge_w_name, merge_b_name],
                                            outputs=[merge_pre_tensor_name],
                                            name=merge_node_name,
                                            alpha=1.0,
                                            beta=1.0,
                                            transB=0,
                                            transA=0)
        merge_layer_relu = onnx.helper.make_node("Relu",
                                                inputs=[merge_pre_tensor_name],
                                                outputs=[merge_pose_tensor_name],
                                                name=merge_relu_node_name)
        
        new_nodes = [split_layer, split_layer_relu, merge_layer, merge_layer_relu]
        new_initializers = [
            onnx.helper.make_tensor(split_w_name, onnx.TensorProto.FLOAT, split_weights.shape, split_weights.flatten().tolist()),
            onnx.helper.make_tensor(split_b_name, onnx.TensorProto.FLOAT, split_bias.shape, split_bias.flatten().tolist()),
            onnx.helper.make_tensor(merge_w_name, onnx.TensorProto.FLOAT, merge_weights.shape, merge_weights.flatten().tolist()),
            onnx.helper.make_tensor(merge_b_name, onnx.TensorProto.FLOAT, merge_bias.shape, merge_bias.flatten().tolist())
        ]

        new_model = self.warpped_model.generate_updated_model(  nodes_to_replace=[gemm_node, relu_node],
                                                                additional_nodes=new_nodes,
                                                                additional_initializers=new_initializers,
                                                                graph_name=f"{self.onnx_path.stem}_split",
                                                                producer_name="ReluSplitter")
        return new_model


    def _split_fc_baseline(self, nodes_to_split, split_idxs):
        device="cuda"if torch.cuda.is_available() else "cpu"

        gemm_node, relu_node = nodes_to_split
        n_splits = len(split_idxs)

        grad, offset = self.warpped_model.get_gemm_wb(gemm_node)    # w,b of the layer to be splitted
        # create new layers
        num_out, num_in = grad.shape
        split_weights = np.zeros((num_out + n_splits, num_in))
        split_bias = np.zeros(num_out + n_splits)
        merge_weights = np.zeros((num_out, num_out + n_splits))
        merge_bias = np.zeros(num_out)

        # update: pick split_idxs randomly
        split_idxs = random.sample(range(num_out), n_splits)
        
        idx = 0     # index of neuron in the new split layer
        for i in range(num_out):
            if i in split_idxs:
                r1 = random.uniform(0, 1)
                r2 = 1 - r1

                w = grad[i]
                b = offset[i]
                
                split_weights[idx]      = w * r1
                split_weights[idx+1]    = w * r2
                split_bias[idx]         = b * r1
                split_bias[idx+1]       = b * r2
                merge_weights[i][idx]   = 1
                merge_weights[i][idx+1] = 1
                merge_bias[i] = 0
                idx += 2
            else:
                split_weights[idx] = grad[i]
                split_bias[idx] = offset[i]
                merge_weights[i][idx] = 1
                idx += 1
        split_weights = split_weights.T
        merge_weights = merge_weights.T
        
        orginal_input_name = gemm_node.input[0]
        orginal_output_name = relu_node.output[0]
        split_pre_tensor_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_sPre")
        split_post_tensor_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_sPost")
        merge_pre_tensor_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_mPre")
        merge_pose_tensor_name = orginal_output_name
        split_w_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_sW")
        split_b_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_sB")
        merge_w_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_mW")
        merge_b_name = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}_mB")
        split_node_name      = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}_s")
        split_relu_node_name = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}_sR")
        merge_node_name      = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}_m")
        merge_relu_node_name = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}_mR")

        # create the new split layer
        split_layer = onnx.helper.make_node("Gemm",
                                            inputs=[orginal_input_name, split_w_name, split_b_name],
                                            outputs=[split_pre_tensor_name],
                                            name=split_node_name,
                                            alpha=1.0,
                                            beta=1.0,
                                            transB=0,
                                            transA=0)
        split_layer_relu = onnx.helper.make_node("Relu",
                                                inputs=[split_pre_tensor_name],
                                                outputs=[split_post_tensor_name],
                                                name=split_relu_node_name)
        # create the new merge layer
        merge_layer = onnx.helper.make_node("Gemm",
                                            inputs=[split_post_tensor_name, merge_w_name, merge_b_name],
                                            outputs=[merge_pre_tensor_name],
                                            name=merge_node_name,
                                            alpha=1.0,
                                            beta=1.0,
                                            transB=0,
                                            transA=0)
        merge_layer_relu = onnx.helper.make_node("Relu",
                                                inputs=[merge_pre_tensor_name],
                                                outputs=[merge_pose_tensor_name],
                                                name=merge_relu_node_name)
        
        new_nodes = [split_layer, split_layer_relu, merge_layer, merge_layer_relu]
        new_initializers = [
            onnx.helper.make_tensor(split_w_name, onnx.TensorProto.FLOAT, split_weights.shape, split_weights.flatten().tolist()),
            onnx.helper.make_tensor(split_b_name, onnx.TensorProto.FLOAT, split_bias.shape, split_bias.flatten().tolist()),
            onnx.helper.make_tensor(merge_w_name, onnx.TensorProto.FLOAT, merge_weights.shape, merge_weights.flatten().tolist()),
            onnx.helper.make_tensor(merge_b_name, onnx.TensorProto.FLOAT, merge_bias.shape, merge_bias.flatten().tolist())
        ]

        new_model = self.warpped_model.generate_updated_model(  nodes_to_replace=[gemm_node, relu_node],
                                                                additional_nodes=new_nodes,
                                                                additional_initializers=new_initializers,
                                                                graph_name=f"{self.onnx_path.stem}_split",
                                                                producer_name="ReluSplitter")
        return new_model