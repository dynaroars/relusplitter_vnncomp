from .common import *

class RSplitter_input():
    # def split_fc(self, nodes_to_split, n_splits=None, split_mask="stable", fc_strategy="random", scale_factors=(1.0, -1.0), create_baseline=False, bounding_method="backward"):


    def input_split_inplace(self, nodes_to_split, split_idxs, ):
        # fc
        pass
    
    def _intput_split_inplace_fc(self, node_to_split, split_idxs, split_ratios):
        # needs to return a vnnlib onnx pair
        pass
        


    def input_split_w_layer():
        # easier to implement
        # use fc or conv depending on the input type
        pass

