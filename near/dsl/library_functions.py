import torch
import torch.nn as nn
import pytorch_lightning as pl

from .neural_functions import init_neural_function, HeuristicNeuralFunction


class LibraryFunction(pl.LightningModule):

    def __init__(self, input_type, output_type, input_size, output_size, num_units, name=""):
        super().__init__()

        self.input_type = input_type
        self.output_type = output_type
        self.input_size = input_size
        self.output_size = output_size
        self.num_units = num_units
        self.name = name

        self.submodules = nn.ModuleDict({})
        self.params = nn.ParameterDict({})
        self.init_params()
        # TODO init_submodules here?

    def init_program(self):
        self.init_params()
        self.init_submodules()

    def init_params(self):
        pass # Override if neccessary

    def init_submodules(self):
        for submod in self.submodules.values():
            if isinstance(submod, HeuristicNeuralFunction):
                submod.init_model()
            elif isinstance(submod, LibraryFunction):
                submod.init_program()
            else:
                raise NotImplementedError

    def has_params(self):
        return len(self.params) > 0

    def get_typesignature(self):
        return self.input_type, self.output_type

    def sizes(self):
        return (self.input_size, self.output_size)

    def to_str(self, include_params=False):
        # Don't want to override default str representation for Modules
        str_items = []

        for submod in self.submodules.values():
            str_items.append(submod.to_str(include_params=include_params))            

        if include_params and self.has_params():
            for name, param in self.params.items():
                str_items.append(f"{name}: {param.data}")

        return f"{self.name}({', '.join(str_items)})"

    def execute_on_batch(self, batch, batch_lens=None):
        # batch has size (batch_size, seq_len, state_dim)
        raise NotImplementedError

class StartFunction(LibraryFunction):

    def __init__(self, input_type, output_type, input_size, output_size, num_units):
        super().__init__(input_type, output_type, input_size, output_size, num_units, name="Start")

        self.submodules["program"] = init_neural_function(input_type, output_type, input_size, output_size, num_units)

    def execute_on_batch(self, batch, batch_lens=None, batch_output=None, is_sequential=False):
        return self.submodules["program"].execute_on_batch(batch, batch_lens)
            
class FoldFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, fold_function=None):
        super().__init__("list", "atom", input_size, output_size, num_units, name="Fold")

        if fold_function is None:
            fold_function = init_neural_function("atom", "atom", input_size+output_size, output_size, num_units)
        self.submodules["foldfunction"] = fold_function

        #TODO: will this accumulator require a grad?
        self.accumulator = torch.zeros(output_size)
        
    def execute_on_batch(self, batch, batch_lens=None, is_sequential=False):
        assert len(batch.size()) == 3
        batch_size, seq_len, feature_dim = batch.size()
        batch = batch.transpose(0,1) # (seq_len, batch_size, feature_dim)

        fold_out = []
        folded_val = self.accumulator.clone().detach().requires_grad_(True)
        folded_val = folded_val.unsqueeze(0).repeat(batch_size,1).to(batch.device)
        for t in range(seq_len):
            features = batch[t]
            out_val = self.submodules["foldfunction"].execute_on_batch(torch.cat([features, folded_val], dim=1))
            fold_out.append(out_val.unsqueeze(1))
            folded_val = out_val
        fold_out = torch.cat(fold_out, dim=1)
        
        if not is_sequential:
            idx = torch.tensor(batch_lens).to(batch.device) - 1
            idx = idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, fold_out.size(-1))
            fold_out = fold_out.gather(1, idx).squeeze(1)

        return fold_out

class MapFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, map_function=None):
        super().__init__("list", "list", input_size, output_size, num_units, name="Map")

        if map_function is None:
            map_function = init_neural_function("atom", "atom", input_size, output_size, num_units)
        self.submodules["mapfunction"] = map_function

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 3
        batch_size, seq_len, feature_dim = batch.size()
        map_input = batch.view(-1, feature_dim)
        map_output = self.submodules["mapfunction"].execute_on_batch(map_input)
        return map_output.view(batch_size, seq_len, -1)

class MapPrefixesFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, map_function=None):
        super().__init__("list", "list", input_size, output_size, num_units, name="MapPrefixes")

        if map_function is None:
            map_function = init_neural_function("list", "atom", input_size, output_size, num_units)
        self.submodules["mapfunction"] = map_function

    def execute_on_batch(self, batch, batch_lens):
        assert len(batch.size()) == 3
        map_output = self.submodules["mapfunction"].execute_on_batch(batch, batch_lens, is_sequential=True)
        assert len(map_output.size()) == 3
        return map_output

class MapAverageFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, a2a_function=None):
        super().__init__("list", "atom", input_size, output_size, num_units, name="MapAverage")

        if a2a_function is None:
            a2a_function = init_neural_function("atom", "atom", input_size, output_size, num_units)
        self.submodules["a2a_function"] = a2a_function
        
    def init_params(self):
        self.params['threshold'] = torch.nn.Parameter(torch.rand(self.output_size))

    def execute_on_batch(self, batch, batch_lens):
        assert len(batch.size()) == 3
        batch_size, seq_len, feature_dim = batch.size()
        batch = batch.transpose(0,1) # (seq_len, batch_size, feature_dim)

        map_output = []
        for t in range(seq_len):
            map_output.append(self.submodules["a2a_function"].execute_on_batch(batch[t]))
        map_output = torch.stack(map_output).mean(dim=0)

        return map_output - self.params['threshold'].to(map_output.device)

class ITE(LibraryFunction):
    """(Smoothed) If-The-Else."""

    def __init__(self, input_type, output_type, input_size, output_size, num_units, eval_function=None, function1=None, function2=None, beta=1.0, name="ITE", simple=False):
        super().__init__(input_type, output_type, input_size, output_size, num_units, name=name)
        
        if eval_function is None:
            if simple:
                eval_function = init_neural_function(input_type, "atom", input_size, 1, num_units)
            else:
                eval_function = init_neural_function(input_type, "atom", input_size, output_size, num_units)
        self.submodules["evalfunction"] = eval_function
        
        if function1 is None:
            function1 = init_neural_function(input_type, output_type, input_size, output_size, num_units)
        self.submodules["function1"] = function1

        if function2 is None:
            function2 = init_neural_function(input_type, output_type, input_size, output_size, num_units)
        self.submodules["function2"] = function2

        self.bsmooth = nn.Sigmoid()
        self.beta = beta
        self.simple = simple # the simple version of ITE evaluates the same function for all dimensions of the output
        
    def execute_on_batch(self, batch, batch_lens=None, is_sequential=False):
        if self.input_type == 'list':
            assert len(batch.size()) == 3
            assert batch_lens is not None
        else:
            assert len(batch.size()) == 2
        if is_sequential:
            predicted_eval = self.submodules["evalfunction"].execute_on_batch(batch, batch_lens, is_sequential=False)
            predicted_function1 = self.submodules["function1"].execute_on_batch(batch, batch_lens, is_sequential=is_sequential)
            predicted_function2 = self.submodules["function2"].execute_on_batch(batch, batch_lens, is_sequential=is_sequential)
        else:
            predicted_eval = self.submodules["evalfunction"].execute_on_batch(batch, batch_lens)
            predicted_function1 = self.submodules["function1"].execute_on_batch(batch, batch_lens)
            predicted_function2 = self.submodules["function2"].execute_on_batch(batch, batch_lens)

        gate = self.bsmooth(predicted_eval*self.beta)
        if self.simple:
            gate = gate.repeat(1, self.output_size)
        
        if self.get_typesignature() == ('list', 'list'):
            gate = gate.unsqueeze(1).repeat(1, batch.size(1), 1)
        elif self.get_typesignature() == ('list', 'atom') and is_sequential:
            gate = gate.unsqueeze(1).repeat(1, batch.size(1), 1)

        assert gate.size() == predicted_function2.size() == predicted_function1.size()
        ite_result = gate*predicted_function1 + (1.0 - gate)*predicted_function2

        return ite_result

class SimpleITE(ITE):
    """The simple version of ITE evaluates one function for all dimensions of the output."""

    def __init__(self, input_type, output_type, input_size, output_size, num_units, eval_function=None, function1=None, function2=None, beta=1.0):
        super().__init__(input_type, output_type, input_size, output_size, num_units, 
            eval_function=eval_function, function1=function1, function2=function2, beta=beta, name="SimpleITE", simple=True)
        
class MultiplyFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, function1=None, function2=None):
        super().__init__("atom", "atom", input_size, output_size, num_units, name="Multiply")

        if function1 is None:
            function1 = init_neural_function("atom", "atom", input_size, output_size, num_units)
        self.submodules["function1"] = function1

        if function2 is None:
            function2 = init_neural_function("atom", "atom", input_size, output_size, num_units)
        self.submodules["function2"] = function2

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        predicted_function1 = self.submodules["function1"].execute_on_batch(batch)
        predicted_function2 = self.submodules["function2"].execute_on_batch(batch)
        return predicted_function1 * predicted_function2

class AddFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, function1=None, function2=None):
        super().__init__("atom", "atom", input_size, output_size, num_units, name="Add")

        if function1 is None:
            function1 = init_neural_function("atom", "atom", input_size, output_size, num_units)
        self.submodules["function1"] = function1

        if function2 is None:
            function2 = init_neural_function("atom", "atom", input_size, output_size, num_units)
        self.submodules["function2"] = function2
        
    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        predicted_function1 = self.submodules["function1"].execute_on_batch(batch)
        predicted_function2 = self.submodules["function2"].execute_on_batch(batch)
        return predicted_function1 + predicted_function2

class ContinueFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, fxn=None):
        super().__init__("atom", "atom", input_size, output_size, num_units, name="")

        if fxn is None:
            fxn = init_neural_function("atom", "atom", input_size, output_size, num_units)
        self.submodules["fxn"] = fxn
        
    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        fxn_out = self.submodules["fxn"].execute_on_batch(batch)
        return fxn_out

class LearnedConstantFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units):
        super().__init__({}, "atom", "atom", input_size, output_size, num_units, name="LearnedConstant")

    def init_params(self):
        self.params['constant'] = torch.nn.Parameter(torch.rand(self.output_size))

    def execute_on_batch(self, batch, batch_lens=None):
        return self.params['constant'].unsqueeze(0).repeat(batch.size(0), 1).to(batch.device)
        
class AffineFunction(LibraryFunction):

    def __init__(self, raw_input_size, selected_input_size, output_size, num_units, name="Affine"):
        self.selected_input_size = selected_input_size
        super().__init__("atom", "atom", raw_input_size, output_size, num_units, name=name)

    def init_params(self):
        self.linear_layer = nn.Linear(self.selected_input_size, self.output_size, bias=True)

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        return self.linear_layer(batch)

class AffineFeatureSelectionFunction(AffineFunction):

    def __init__(self, input_size, output_size, num_units, name="AffineFeatureSelection",
        compute_function = None):
        assert hasattr(self, "full_feature_dim")
        assert input_size >= self.full_feature_dim
        if self.full_feature_dim == 0:
            self.is_full = True
            self.full_feature_dim = input_size
        else:
            self.is_full = False
        additional_inputs = input_size - self.full_feature_dim

        self.compute_function = compute_function

        assert hasattr(self, "feature_tensor")
        assert len(self.feature_tensor) <= input_size
        self.feature_tensor = self.feature_tensor
        super().__init__(raw_input_size=input_size, selected_input_size=self.feature_tensor.size()[-1]+additional_inputs, 
            output_size=output_size, num_units=num_units, name=name)

        self.raw_input_size = self.input_size
        if self.is_full:
            self.full_feature_dim = self.input_size
            self.feature_tensor = torch.arange(self.input_size)

        additional_inputs = self.raw_input_size - self.full_feature_dim
        self.selected_input_size = self.feature_tensor.size()[-1] + additional_inputs
        self.linear_layer = nn.Linear(self.selected_input_size, self.output_size, bias=True)

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        if self.compute_function is None:
            features = torch.index_select(batch, 1, self.feature_tensor.to(batch.device))
        else:
            features = self.compute_function(batch)
        remaining_features = batch[:,self.full_feature_dim:]

        self.linear_layer.to(batch.device)

        return self.linear_layer(torch.cat([features, remaining_features], dim=-1))

class FullInputAffineFunction(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = 0 # this will indicate additional_inputs = 0 in FeatureSelectionFunction
        self.feature_tensor = torch.arange(input_size) # selects all features by default
        super().__init__(input_size, output_size, num_units, name="FullFeatureSelect")
