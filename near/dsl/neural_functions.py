import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def init_neural_function(input_type, output_type, input_size, output_size, num_units):
    if (input_type, output_type) == ("list", "list"):
        return ListToListModule(input_size, output_size, num_units)
    elif (input_type, output_type) == ("list", "atom"):
        return ListToAtomModule(input_size, output_size, num_units)
    elif (input_type, output_type) == ("atom", "atom"):
        return AtomToAtomModule(input_size, output_size, num_units)
    else:
        raise NotImplementedError


class HeuristicNeuralFunction(pl.LightningModule):

    def __init__(self, input_type, output_type, input_size, output_size, num_units, name):
        super().__init__()
        self.input_type = input_type
        self.output_type = output_type
        self.input_size = input_size
        self.output_size = output_size
        self.num_units = num_units
        self.name = name

        self.init_model()

    def init_model(self):
        raise NotImplementedError

    def to_str(self, include_params=False):
        # Don't want to override default str representation for Modules
        return self.name

class ListToListModule(HeuristicNeuralFunction):

    def __init__(self, input_size, output_size, num_units):
        super().__init__("list", "list", input_size, output_size, num_units, "ListToListModule")

    def init_model(self):
        self.model = RNNModule(self.input_size, self.output_size, self.num_units)

    def execute_on_batch(self, batch, batch_lens):
        assert len(batch.size()) == 3
        model_out = self.model(batch, batch_lens)
        assert len(model_out.size()) == 3
        return model_out

class ListToAtomModule(HeuristicNeuralFunction):

    def __init__(self, input_size, output_size, num_units):
        super().__init__("list", "atom", input_size, output_size, num_units, "ListToAtomModule")

    def init_model(self):
        self.model = RNNModule(self.input_size, self.output_size, self.num_units)

    def execute_on_batch(self, batch, batch_lens, is_sequential=False):
        assert len(batch.size()) == 3
        model_out = self.model(batch, batch_lens)
        assert len(model_out.size()) == 3

        if not is_sequential:
            idx = torch.tensor(batch_lens).to(batch.device) - 1
            idx = idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, model_out.size(-1))
            model_out = model_out.gather(1, idx).squeeze(1)

        return model_out

class AtomToAtomModule(HeuristicNeuralFunction):

    def __init__(self, input_size, output_size, num_units):
        super().__init__("atom", "atom", input_size, output_size, num_units, "AtomToAtomModule")

    def init_model(self):
        self.model = FeedForwardModule(self.input_size, self.output_size, self.num_units)

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        self.model = self.model.to(batch.device) # TODO weird bug, usually this shouldn't be needed
        model_out = self.model(batch)
        assert len(model_out.size()) == 2
        return model_out


##############################
####### NEURAL MODULES #######
##############################


class RNNModule(pl.LightningModule):

    def __init__(self, input_size, output_size, num_units, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.rnn_size = num_units
        self.num_layers = num_layers
        self.rnn = nn.GRU(self.input_size, self.rnn_size, num_layers=self.num_layers)
        self.out_layer = nn.Linear(self.rnn_size, self.output_size)

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.rnn_size, requires_grad=True)

    def forward(self, batch, batch_lens):
        assert isinstance(batch, torch.Tensor)
        batch_size, seq_len, feature_dim = batch.size()

        # pass through rnn
        hidden = self.init_hidden(batch_size).to(batch.device)
        batch_packed = torch.nn.utils.rnn.pack_padded_sequence(batch, batch_lens, batch_first=True, enforce_sorted=False)
        self.rnn.flatten_parameters()
        out, hidden = self.rnn(batch_packed, hidden)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # pass through linear layer
        out = out.contiguous()
        out = out.view(-1, out.shape[2])
        out = self.out_layer(out)
        out = out.view(batch_size, seq_len, -1)

        return out

class FeedForwardModule(pl.LightningModule):

    def __init__(self, input_size, output_size, num_units):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = num_units
        self.first_layer = nn.Linear(self.input_size, self.hidden_size)
        self.out_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, current_input):
        assert isinstance(current_input, torch.Tensor)
        current = F.relu(self.first_layer(current_input))
        current = self.out_layer(current)
        return current
