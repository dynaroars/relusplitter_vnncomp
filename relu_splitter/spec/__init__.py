import logging
from pathlib import Path
from functools import reduce

from ..utils.read_vnnlib import read_vnnlib

default_logger = logging.getLogger(__name__)


class warpped_vnnlib():
    def __init__(self, vnnlib_rep, model_input_shape=None, model_output_shape=None, logger=default_logger):
        # vnnlib_rep is a list containing 2-tuples:
        #     1. input ranges (box), list of pairs for each input variable
        #     2. specification, provided as a list of pairs (mat, rhs), as in: mat * y <= rhs, where y is the output.
        #                       Each element in the list is a term in a disjunction for the specification.
        self.logger = logger

        # TODO: get input and output prefix
        self.input_prefix = "X"
        self.output_prefix = "Y"

        # inputs
        self.input_spec = vnnlib_rep
        input_bounds, output_bounds = self.input_spec

        self.input_lb = [i[0] for i in input_bounds]
        self.input_ub = [i[1] for i in input_bounds]
        assert all([ lb <= ub for lb, ub in zip(self.input_lb, self.input_ub)]), "Input lower bound is greater than upper bound"

        # self.spec_num_inputs = reduce(lambda x, y: x*y, self.input_lb.shape)
        # if model_input_shape is not None:
        #     assert self.spec_num_inputs == reduce(lambda x, y: x*y, model_input_shape), f"Spec number of inputs does not match model inputs {self.spec_num_inputs} != {reduce(lambda x, y: x*y, self.input_shape)}"
        # else:
        #     self.logger.warning("Input shape is not specified, not checking # input in vnnlib file...")


        # outputs
        self.output_specs = vnnlib_rep[1]
        assert len(self.output_specs) >= 1, "No output specification found"
        # output_specs[disjunction][0:mat, 1:rhs][row of mat]
        self.spec_num_outputs = len(self.output_specs[0][0][0])
        # if model_output_shape is not None:
        #     assert self.spec_num_outputs == reduce(lambda x, y: x*y, model_output_shape), f"Spec number of outputs does not match model outputs {self.spec_num_outputs} != {reduce(lambda x, y: x*y, model_output_shape)}"
        # else:
        #     self.logger.warning("Output shape is not specified, not checking # output in vnnlib file...")

        


    @property
    def bounded_tensor(self):
        if self._bounded_tensor is None:
            # TODO: convert to nhwc
            self._bounded_tensor = BoundedTensor(torch.tensor([self.input_lb]), PerturbationLpNorm(x_L=torch.tensor([self.input_lb]), x_U=torch.tensor([self.input_ub])))
        return self._bounded_tensor


    def serialize_preCondition(self, prefix=None):
        pass
    
    def matNrhs_to_str(self, mat, rhs, prefix=None):
        return "TODO"

    def serialize_postCondition(self, prefix=None):
        if prefix is None:
            prefix = self.output_prefix
        
        # declare output variables
        # (declare-const Y_4 Real)
        output_vars = [f"(declare-const {prefix}_{i} Real)" for i in range(self.spec_num_outputs)]
        var_declarations = "\n".join(output_vars)
        # declare output constraints
        # (assert (<= (+ (* 1.000000 Y_4) (* -1.000000 Y_5)) 0.000000))
        disjunctions = [self.matNrhs_to_str(mat, rhs) for mat, rhs in self.output_specs]
        assertions = ""
        if len(disjunctions) > 1:
            assertions = "(assert (or \n" + '\n\t'.join(disjunctions) + "))"

        else:
            assertions = f"(assert {disjunctions[0]})"

        return f"{var_declarations}\n{assertions}\n"
        



    def serialize(self, kwargs):
        return self.serialize_preCondition(**kwargs) + self.serialize_postCondition(**kwargs)
    
        

    def save(self, fname:Path):
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True)
        with open(fname, "w") as f:
            f.write(self.serialize())
        return fname



    def init_vnnlib(self):
        input_bound, output_bound = read_vnnlib(str(self.spec_path))[0]
        input_lb, input_ub = torch.tensor([[[[i[0] for i in input_bound]]]]), torch.tensor([[[[i[1] for i in input_bound]]]])
        spec_num_inputs = reduce(lambda x, y: x*y, input_lb.shape)
        model_num_inputs = self.warpped_model.num_inputs
        # reshape input bounds to match the model input shape
        input_lb, input_ub = input_lb.view(self.input_shape), input_ub.view(self.input_shape)
        assert torch.all(input_lb <= input_ub), "Input lower bound is greater than upper bound"
        assert model_num_inputs == spec_num_inputs, f"Spec number of inputs does not match model inputs {spec_num_inputs} != {model_num_inputs}"
        self.bounded_input = BoundedTensor(input_lb, PerturbationLpNorm(x_L=input_lb, x_U=input_ub))