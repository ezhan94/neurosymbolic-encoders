import torch
from .library_functions import LibraryFunction


class FinalXPosition(LibraryFunction):

    def __init__(self, input_size, output_size, num_units):
        assert output_size == 1
        super().__init__("list", "atom", input_size, output_size, num_units, name="FinalXPos")

    def init_params(self):
        self.params['threshold'] = torch.nn.Parameter(torch.rand(self.output_size))
        self.params['scale'] = torch.nn.Parameter(torch.rand(self.output_size))

    def execute_on_batch(self, batch, batch_lens):
        assert len(batch.size()) == 3
        final_x = batch[:,-1,0].unsqueeze(-1) # ind 0 is x_pos
        return self.params['scale'].to(batch.device)*final_x - self.params['threshold'].to(batch.device)

class FinalYPosition(LibraryFunction):

    def __init__(self, input_size, output_size, num_units):
        assert output_size == 1
        super().__init__("list", "atom", input_size, output_size, num_units, name="FinalYPos")

    def init_params(self):
        self.params['threshold'] = torch.nn.Parameter(torch.rand(self.output_size))
        self.params['scale'] = torch.nn.Parameter(torch.rand(self.output_size))

    def execute_on_batch(self, batch, batch_lens):
        assert len(batch.size()) == 3
        final_y = batch[:,-1,1].unsqueeze(-1) # ind 1 is y_pos
        return self.params['scale'].to(batch.device)*final_y - self.params['threshold'].to(batch.device)

class AvgSpeed(LibraryFunction):

    def __init__(self, input_size, output_size, num_units):
        assert output_size == 1
        super().__init__("list", "atom", input_size, output_size, num_units, name="AvgSpeed")

    def init_params(self):
        self.params['threshold'] = torch.nn.Parameter(torch.rand(self.output_size))
        self.params['scale'] = torch.nn.Parameter(torch.rand(self.output_size))

    def execute_on_batch(self, batch, batch_lens):
        assert len(batch.size()) == 3
        vel = batch[:,:,-2:] # last 2 dims are velocities
        avg_speed = vel.norm(dim=-1).mean(dim=1).unsqueeze(-1) 
        return self.params['scale'].to(batch.device)*avg_speed - self.params['threshold'].to(batch.device)

class AvgAccel(LibraryFunction):

    def __init__(self, input_size, output_size, num_units):
        assert output_size == 1
        super().__init__("list", "atom", input_size, output_size, num_units, name="AvgAccel")

    def init_params(self):
        self.params['threshold'] = torch.nn.Parameter(torch.rand(self.output_size))
        self.params['scale'] = torch.nn.Parameter(torch.rand(self.output_size))

    def execute_on_batch(self, batch, batch_lens):
        assert len(batch.size()) == 3
        vel = batch[:,:,-2:] # last 2 dims are velocities
        accel = vel[:,1:] - vel[:,:-1]
        avg_accel = accel.norm(dim=-1).mean(dim=1).unsqueeze(-1)
        return self.params['scale'].to(batch.device)*avg_accel - self.params['threshold'].to(batch.device)
