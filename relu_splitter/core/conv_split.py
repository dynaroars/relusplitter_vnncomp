from .common import *

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8" if needed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.enabled = False


class RSplitter_conv():
    def conv_get_split_masks(self, layer_bounds):
        output_lb, output_ub = layer_bounds
        masks = []
        for i in range(output_lb.shape[0]):
        # for i in range(output_lb.shape[1]):
            # lb, ub = output_lb[0,i], output_ub[0,i]
            lb, ub = output_lb[i], output_ub[i]
            temp = {}
            temp["all"] = torch.ones_like(lb)
            temp["stable"] = torch.logical_or(lb >= 0, ub <= 0)
            temp["unstable"] = ~temp["stable"]
            temp["stable+"] = lb >= 0
            temp["stable-"] = ub <= 0
            temp["unstable_n_stable+"] = torch.logical_or(temp["unstable"], temp["stable+"])
            assert torch.all(temp["stable"] == (temp["stable+"] | temp["stable-"])), "stable is not equal to stable+ AND stable-"
            assert not torch.any(temp["stable+"] & temp["stable-"]), "stable+ and stable- are not mutually exclusive"
            assert not torch.any(temp["unstable"] & temp["stable"]), "unstable and stable are not mutually exclusive"
            assert torch.all((temp["unstable"] | temp["stable"]) == temp["all"]), "The union of unstable and stable does not cover all elements"
            masks.append(temp)
        return masks

    def conv_compute_split_layer_bias(self, kernel_bounds, mask, strategy):
        if not torch.any(mask):
            self.logger.warn("no patch to be considered for bias computation, using 0 as default bias...")
            return 0.0
        
        kernel_lb, kernel_ub = kernel_bounds
        self.logger.debug(f"Kernel bound shapes: {kernel_lb.shape}, {kernel_ub.shape}")
        self.logger.debug(f"Mask shape: {mask.shape}")
        # only keep the unmasked values
        kernel_lb, kernel_ub, mask = kernel_lb.flatten(), kernel_ub.flatten(), mask.flatten()
        lb, ub = kernel_lb[mask], kernel_ub[mask]
        # find the val to sat most intervals using sliding line sweep
        evnts = []
        for i in range(len(lb)):
            evnts.append((lb[i].item(), 1))
            evnts.append((ub[i].item(), -1))
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

        # return a random value from the range
        bias = random.uniform(start, end)  
        self.logger.debug(f"Max active patches / # patches considered / # patches")
        self.logger.debug(f"{max_active_intervals} / {len(lb)} / {len(kernel_lb)}")
        self.logger.debug(f"Max active patches range: [{start}, {end}]")
        self.logger.debug(f"Selected bias: {bias}")
        return bias

    def conv_compute_split_layer_bias_optimized(self, kernel_bounds, mask, strategy):
        if not torch.any(mask):
            self.logger.warn("no patch to be considered for bias computation, using 0 as default bias...")
            return 0.0
        
        kernel_lb, kernel_ub = kernel_bounds
        self.logger.debug(f"Kernel bound shapes: {kernel_lb.shape}, {kernel_ub.shape}")
        self.logger.debug(f"Mask shape: {mask.shape}")
        # only keep the unmasked values
        # kernel_lb, kernel_ub, mask = kernel_lb.flatten(), kernel_ub.flatten(), mask.flatten()
        # lb, ub = kernel_lb[mask], kernel_ub[mask]
        lb, ub = kernel_lb.flatten(), kernel_ub.flatten()
        # find the val to sat most intervals using sliding line sweep
        evnts = []
        for i in range(len(lb)):
            evnts.append((lb[i].item(), 1))
            evnts.append((ub[i].item(), -1))
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

        # return a random value from the range
        bias = random.uniform(start, end)  
        self.logger.debug(f"Max active patches / # patches considered / # patches")
        self.logger.debug(f"{max_active_intervals} / {len(lb)} / {len(kernel_lb)}")
        self.logger.debug(f"Max active patches range: [{start}, {end}]")
        self.logger.debug(f"Selected bias: {bias}")
        return bias



    def split_conv(self, nodes_to_split, n_splits=None, split_mask="stable", conv_strategy="max_unstable", scale_factors=(1.0, -1.0), create_baseline=False, bounding_method="backward", bias_method="normal"):
        conv_node, relu_node = nodes_to_split
        assert conv_node.op_type == "Conv" and relu_node.op_type == "Relu"

        self.logger.debug("=====================================")     
        self.logger.debug(f"Splitting model: {self.onnx_path} with spec: {self.spec_path}")
        self.logger.debug(f"Splitting at Conv node: <{conv_node.name}> && ReLU node: <{relu_node.name}>")
        self.logger.debug(f"Split mask: {split_mask}")
        self.logger.debug(f"Scale factors: {scale_factors}")

        ori_model = self.warpped_model
        ori_w, ori_b = ori_model.get_conv_wb(conv_node)
        assert ori_w.dim() == 4
        ori_oC, ori_iC, ori_kH, ori_kW = ori_w.shape

        # prepare the bounds
        layer_lb,layer_ub = self.warpped_model.get_bound_of(self.bounded_input, conv_node.output[0], method=bounding_method)
        # remove single dimension - lli: adhoc fix!!!!
        layer_lb, layer_ub = layer_lb.squeeze(0), layer_ub.squeeze(0)

        masks = self.conv_get_split_masks((layer_lb, layer_ub))
        # decide kernels to split
        split_idxs = []
        n_splits = 16   # For vnncomp again, cap number of conv splits to 16
        if n_splits == None or n_splits >= ori_oC:
            split_idxs = list(range(ori_oC))
            self.logger.info(f"Splitting all kernels")
        else:
            split_idxs = random.sample(range(ori_oC), n_splits)
            self.logger.info(f"Randomly selected kernels: {split_idxs}")
        self.logger.debug(f"Conv layer bound computed, shapes: {layer_lb.shape}, {layer_ub.shape}")
        # compute the split bias for each kernel
        split_biases = []

        for i in split_idxs:
            # adhoc testing TODO
            if bias_method == "normal":
                split_biases.append(self.conv_compute_split_layer_bias((layer_lb[i], layer_ub[i]), masks[i][split_mask], conv_strategy))
            elif bias_method == "optimized":
                split_biases.append(self.conv_compute_split_layer_bias_optimized((layer_lb[i], layer_ub[i]), masks[i][split_mask], conv_strategy))
            # split_biases.append(self.conv_compute_split_layer_bias((layer_lb[i], layer_ub[i]), masks[i][split_mask], conv_strategy))
            # split_biases.append(self.conv_compute_split_layer_bias((layer_lb[0,i], layer_ub[0,i]), masks[i][split_mask], conv_strategy))
            self.logger.info(f"Selected split bias for kernel {i}: {split_biases[-1]}")

        # Create the models
        split_model = self._split_conv(nodes_to_split, split_idxs, split_biases, scale_factors)
        baseline_model = self._split_conv_baseline(nodes_to_split, split_idxs) if create_baseline else None

        return (split_model, baseline_model)


    def _split_conv(self, nodes_to_split, split_idxs=[], split_biases=[], scale_factors=(1.0, -1.0)):
        conv_node, relu_node = nodes_to_split
        ori_model = self.warpped_model

        # get the original conv layer attributes
        ori_input = conv_node.input[0]
        ori_output = relu_node.output[0]
        ori_graph_name = ori_model._model.graph.name
        ori_groups = conv_node.attribute[0].i
        ori_dilations = conv_node.attribute[1].ints
        if ori_dilations == []:
            ori_dilations = [1,1]
        ori_kernel_shape = conv_node.attribute[2].ints
        ori_pads = conv_node.attribute[3].ints
        ori_strides = conv_node.attribute[4].ints

        # get the original weights and bias
        K_o, b_o = ori_model.get_conv_wb(conv_node)
        assert K_o.dim() == 4
        ori_oC, ori_iC, ori_kH, ori_kW = K_o.shape

        # make weights and bias for split and merge layers
        num_splits = len(split_idxs)
        new_oC = ori_oC + num_splits

        new_w = np.zeros( (new_oC, ori_iC, ori_kH, ori_kW))
        new_b = np.zeros( (new_oC,) )
        merge_w = np.zeros( (ori_oC, new_oC, 1, 1) )
        merge_b = np.zeros( (ori_oC,) )

        curr_idx = 0
        scale_ratio_pos, scale_ratio_neg = scale_factors
        for i in tqdm(range(ori_oC), desc="Constructing new Conv layer..."):
            if i in split_idxs:
                split_offset = split_biases[split_idxs.index(i)]
                self.logger.info(f"Splitting kernel {i} with bias: {split_offset}")
                # split the kernel
                new_w[curr_idx]     =  scale_ratio_pos * K_o[i]
                new_w[curr_idx+1]   =  scale_ratio_neg * K_o[i]
                new_b[curr_idx]     =  -scale_ratio_pos * (split_offset - b_o[i])
                new_b[curr_idx+1]   =  -scale_ratio_neg * (split_offset - b_o[i])
                # set merge w & b
                merge_w[i, curr_idx  , 0, 0] = 1/scale_ratio_pos
                merge_w[i, curr_idx+1, 0, 0] = 1/scale_ratio_neg
                merge_b[i] = split_offset

                curr_idx += 2
            else:
                # copy the kernel
                new_w[curr_idx] = K_o[i]
                new_b[curr_idx] = b_o[i]
                # forward through merge layer
                merge_w[i, curr_idx, 0, 0] = 1
                merge_b[i] = 0
                curr_idx += 1
        # compose the new onnx nodes
        # tensor names
        input_tensor_name       = ori_input
        output_tensor_name      = ori_output
        split_preRelu_tensor_name   = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_sPre")
        split_postRelu_tensor_name  = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_sPost")
        merge_preRelu_tensor_name   = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_mPre")
        merge_postRelu_tensor_name  = output_tensor_name
        # initializer names
        split_w_init_name       = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_sW")
        split_b_init_name       = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_sB")
        merge_w_init_name       = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_mW")
        merge_b_init_name       = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_mB")
        # node names
        split_node_name         = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}[Conv]_s")
        split_relu_node_name    = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}[Conv]_sReLU")
        merge_node_name         = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}[Conv]_m")
        merge_relu_node_name    = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}[Conv]_mReLU")

        split_conv_layer = onnx.helper.make_node(
            'Conv',
            inputs=[input_tensor_name, split_w_init_name, split_b_init_name],
            outputs=[split_preRelu_tensor_name],
            name=split_node_name,
            # group=ori_groups,
            dilations=ori_dilations,
            kernel_shape=ori_kernel_shape,
            pads=ori_pads,
            strides=ori_strides
        )
        split_relu_layer = onnx.helper.make_node(
            'Relu',
            inputs=[split_preRelu_tensor_name],
            outputs=[split_postRelu_tensor_name],
            name=split_relu_node_name
        )
        merge_conv_layer = onnx.helper.make_node(
            'Conv',
            inputs=[split_postRelu_tensor_name, merge_w_init_name, merge_b_init_name],
            outputs=[merge_preRelu_tensor_name],
            name=merge_node_name,
            # group=ori_groups,
            dilations=[1,1],
            kernel_shape=[1,1],
            pads=[0,0,0,0],
            strides=[1,1]
        )
        merge_relu_layer = onnx.helper.make_node(
            'Relu',
            inputs=[merge_preRelu_tensor_name],
            outputs=[merge_postRelu_tensor_name],
            name=merge_relu_node_name
        )

        new_nodes = [split_conv_layer, split_relu_layer, merge_conv_layer, merge_relu_layer]
        new_initializers = [
            onnx.helper.make_tensor(split_w_init_name, onnx.TensorProto.FLOAT, new_w.shape, new_w.flatten()),
            onnx.helper.make_tensor(split_b_init_name, onnx.TensorProto.FLOAT, new_b.shape, new_b.flatten()),
            onnx.helper.make_tensor(merge_w_init_name, onnx.TensorProto.FLOAT, merge_w.shape, merge_w.flatten()),
            onnx.helper.make_tensor(merge_b_init_name, onnx.TensorProto.FLOAT, merge_b.shape, merge_b.flatten())
        ]

        new_model = self.warpped_model.generate_updated_model(  nodes_to_replace=[conv_node, relu_node],
                                                                additional_nodes=new_nodes,
                                                                additional_initializers=new_initializers,
                                                                graph_name=ori_graph_name,
                                                                producer_name="ReluSplitter")
        return new_model



    def _split_conv_baseline(self, nodes_to_split, split_idxs=[]):
            conv_node, relu_node = nodes_to_split
            ori_model = self.warpped_model

            # get the original conv layer attributes
            ori_input = conv_node.input[0]
            ori_output = relu_node.output[0]
            ori_graph_name = ori_model._model.graph.name
            ori_groups = conv_node.attribute[0].i
            ori_dilations = conv_node.attribute[1].ints
            if ori_dilations == []:
                ori_dilations = [1,1]
            ori_kernel_shape = conv_node.attribute[2].ints
            ori_pads = conv_node.attribute[3].ints
            ori_strides = conv_node.attribute[4].ints

            # get the original weights and bias
            ori_w, ori_b = ori_model.get_conv_wb(conv_node)
            assert ori_w.dim() == 4
            ori_oC, ori_iC, ori_kH, ori_kW = ori_w.shape

            # make weights and bias for split and merge layers
            n_splits = len(split_idxs)
            new_oC = ori_oC + n_splits

            new_w = np.zeros( (new_oC, ori_iC, ori_kH, ori_kW))
            new_b = np.zeros( (new_oC,) )
            merge_w = np.zeros( (ori_oC, new_oC, 1, 1) )
            merge_b = np.zeros( (ori_oC,) )

            # update: pick split_idxs randomly
            split_idxs = random.sample(range(ori_oC), n_splits)

            curr_idx = 0
            for i in tqdm(range(ori_oC), desc="Constructing new Conv layer..."):
                if i in split_idxs:
                    r1 = random.uniform(0, 1)
                    r2 = 1 - r1

                    temp_w = ori_w[i]
                    temp_b = ori_b[i]

                    self.logger.info(f"Splitting kernel {i}...")
                    # split the kernel
                    new_w[curr_idx]     = temp_w * r1
                    new_w[curr_idx+1]   = temp_w * r2
                    new_b[curr_idx]     = temp_b * r1
                    new_b[curr_idx+1]   = temp_b * r2
                    # set merge w & b
                    merge_w[i, curr_idx  , 0, 0] = 1
                    merge_w[i, curr_idx+1, 0, 0] = 1
                    merge_b[i] = 0

                    curr_idx += 2
                else:
                    # copy the kernel
                    new_w[curr_idx] = ori_w[i]
                    new_b[curr_idx] = ori_b[i]
                    # forward through merge layer
                    merge_w[i, curr_idx, 0, 0] = 1
                    merge_b[i] = 0
                    curr_idx += 1
            # compose the new onnx nodes
            # tensor names
            input_tensor_name       = ori_input
            output_tensor_name      = ori_output
            split_preRelu_tensor_name   = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_sPre")
            split_postRelu_tensor_name  = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_sPost")
            merge_preRelu_tensor_name   = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_mPre")
            merge_postRelu_tensor_name  = output_tensor_name
            # initializer names
            split_w_init_name       = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_sW")
            split_b_init_name       = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_sB")
            merge_w_init_name       = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_mW")
            merge_b_init_name       = self.warpped_model.gen_tensor_name(prefix=f"{TOOL_NAME}[Conv]_mB")
            # node names
            split_node_name         = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}[Conv]_s")
            split_relu_node_name    = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}[Conv]_sReLU")
            merge_node_name         = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}[Conv]_m")
            merge_relu_node_name    = self.warpped_model.gen_node_name(prefix=f"{TOOL_NAME}[Conv]_mReLU")

            split_conv_layer = onnx.helper.make_node(
                'Conv',
                inputs=[input_tensor_name, split_w_init_name, split_b_init_name],
                outputs=[split_preRelu_tensor_name],
                name=split_node_name,
                # group=ori_groups,
                dilations=ori_dilations,
                kernel_shape=ori_kernel_shape,
                pads=ori_pads,
                strides=ori_strides
            )
            split_relu_layer = onnx.helper.make_node(
                'Relu',
                inputs=[split_preRelu_tensor_name],
                outputs=[split_postRelu_tensor_name],
                name=split_relu_node_name
            )
            merge_conv_layer = onnx.helper.make_node(
                'Conv',
                inputs=[split_postRelu_tensor_name, merge_w_init_name, merge_b_init_name],
                outputs=[merge_preRelu_tensor_name],
                name=merge_node_name,
                # group=ori_groups,
                dilations=[1,1],
                kernel_shape=[1,1],
                pads=[0,0,0,0],
                strides=[1,1]
            )
            merge_relu_layer = onnx.helper.make_node(
                'Relu',
                inputs=[merge_preRelu_tensor_name],
                outputs=[merge_postRelu_tensor_name],
                name=merge_relu_node_name
            )

            new_nodes = [split_conv_layer, split_relu_layer, merge_conv_layer, merge_relu_layer]
            new_initializers = [
                onnx.helper.make_tensor(split_w_init_name, onnx.TensorProto.FLOAT, new_w.shape, new_w.flatten()),
                onnx.helper.make_tensor(split_b_init_name, onnx.TensorProto.FLOAT, new_b.shape, new_b.flatten()),
                onnx.helper.make_tensor(merge_w_init_name, onnx.TensorProto.FLOAT, merge_w.shape, merge_w.flatten()),
                onnx.helper.make_tensor(merge_b_init_name, onnx.TensorProto.FLOAT, merge_b.shape, merge_b.flatten())
            ]

            new_model = self.warpped_model.generate_updated_model(  nodes_to_replace=[conv_node, relu_node],
                                                                    additional_nodes=new_nodes,
                                                                    additional_initializers=new_initializers,
                                                                    graph_name=ori_graph_name,
                                                                    producer_name="ReluSplitter_baseline")
            return new_model