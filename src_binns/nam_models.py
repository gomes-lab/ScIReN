"""
Models described in the Neural Additive Models paper.

Source: https://github.com/kherud/neural-additive-models-pt/blob/master/nam/model.py
"""
from typing import Union, Iterable, Sized, Tuple
import torch
import torch.nn.functional as F


def truncated_normal_(tensor, mean: float = 0., std: float = 1.):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class ActivationLayer(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((in_features, out_features)))
        self.bias = torch.nn.Parameter(torch.empty(in_features))

    def forward(self, x):
        raise NotImplementedError("abstract method called")


class ExULayer(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        truncated_normal_(self.weight, mean=4.0, std=0.5)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        exu = (x - self.bias) @ torch.exp(self.weight)
        return torch.clip(exu, 0, 1)


class ReLULayer(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        torch.nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        return F.relu((x - self.bias) @ self.weight)


class FeatureNN(torch.nn.Module):
    def __init__(self,
                 shallow_units: int,
                 hidden_units: Tuple = (),
                 shallow_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,
                 dropout: float = .5,
                 n_outputs: int = 1,
                 ):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            hidden_layer(shallow_units if i == 0 else hidden_units[i - 1], hidden_units[i])
            for i in range(len(hidden_units))
        ])
        self.layers.insert(0, shallow_layer(1, shallow_units))

        self.dropout = torch.nn.Dropout(p=dropout)

        self.linear = torch.nn.Linear(shallow_units if len(hidden_units) == 0 else hidden_units[-1], n_outputs, bias=False)
        torch.nn.init.xavier_uniform_(self.linear.weight)


    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        return self.linear(x)


class NeuralAdditiveModel(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 shallow_units: int,
                 hidden_units: Tuple = (),
                 shallow_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,
                 feature_dropout: float = 0.,
                 hidden_dropout: float = 0.,
                 ):
        super().__init__()
        self.input_size = input_size

        if isinstance(shallow_units, list):
            assert len(shallow_units) == input_size
        elif isinstance(shallow_units, int):
            shallow_units = [shallow_units for _ in range(input_size)]

        self.feature_nns = torch.nn.ModuleList([
            FeatureNN(shallow_units=shallow_units[i],
                      hidden_units=hidden_units,
                      shallow_layer=shallow_layer,
                      hidden_layer=hidden_layer,
                      dropout=hidden_dropout)
            for i in range(input_size)
        ])
        self.feature_dropout = torch.nn.Dropout(p=feature_dropout)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        f_out = torch.cat(self._feature_nns(x), dim=-1)
        f_out = self.feature_dropout(f_out)
        self.f_out = f_out
        return f_out.sum(axis=-1) + self.bias, f_out

    def _feature_nns(self, x):
        return [self.feature_nns[i](x[:, i]) for i in range(self.input_size)]



class MultiOutputNAM(torch.nn.Module):
    """
    Same as NeuralAdditiveModel, but there is a separate network for each input-output pair.

    input_size is the number of NUMERIC input features, and n_outputs is the number of outputs.
    
    The actual input will contain (input_size + n_categorical*n_outputs) features: each
    categorical feature has already been passed through an Embedding layer that provides
    an additive contribution from each category for each output.
    """
    def __init__(self,
                 input_size: int,
                 shallow_units: int,
                 hidden_units: Tuple = (),
                 shallow_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,
                 feature_dropout: float = 0.,
                 hidden_dropout: float = 0.,
                 n_outputs: int = 1,
                 ):
        super().__init__()
        self.input_size = input_size
        self.n_outputs = n_outputs

        if isinstance(shallow_units, list):
            assert len(shallow_units) == input_size
        elif isinstance(shallow_units, int):
            shallow_units = [shallow_units for _ in range(input_size * n_outputs)]

        self.feature_nns = torch.nn.ModuleList([
            FeatureNN(shallow_units=shallow_units[i],
                      hidden_units=hidden_units,
                      shallow_layer=shallow_layer,
                      hidden_layer=hidden_layer,
                      dropout=hidden_dropout)
            for i in range(input_size * n_outputs)
        ])
        self.feature_dropout = torch.nn.Dropout(p=feature_dropout)
        self.bias = torch.nn.Parameter(torch.zeros(self.n_outputs))

    def forward(self, x):
        x_numeric = x[:, :self.input_size]
        x_categorical = x[:, self.input_size:]
        f_out = self._feature_nns(x_numeric)  # List of (n_features*n_outputs) tensors of shape [batch, 1]
        f_out = torch.cat(f_out + [x_categorical], dim=-1)  # [batch, n_features*n_outputs], n_features includes both numeric + categorical features
        f_out = f_out.reshape((f_out.shape[0], -1, self.n_outputs))  # [batch, n_features, n_outputs]

        # Divide by a constant to ensure that the summed output at initialization does not
        # saturate the sigmoid. This constant could be tuned. Theoretically should it be sqrt(n_features)?
        self.divide_by = f_out.shape[1]  # n_features
        f_out = f_out / (self.divide_by ** 2)  # TEMP 
        f_out = self.feature_dropout(f_out)
        self.f_out = f_out
        return f_out.sum(axis=1) + self.bias

    def _feature_nns(self, x):
        outputs = []
        for i in range(self.input_size):
            for j in range(self.n_outputs):
                single_output = self.feature_nns[self.n_outputs * i + j](x[:, i])
                outputs.append(single_output)
        return outputs


class MultiOutputJointNAM(torch.nn.Module):
    """
    Same as NeuralAdditiveModel, but each feature-network produces multiple outputs.

    input_size is the number of NUMERIC input features, and n_outputs is the number of outputs.
    
    The actual input will contain (input_size + n_categorical*n_outputs) features: each
    categorical feature has already been passed through an Embedding layer that provides
    an additive contribution from each category for each output.
    """
    def __init__(self,
                 input_size: int,
                 shallow_units: int,
                 hidden_units: Tuple = (),
                 shallow_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,
                 feature_dropout: float = 0.,
                 hidden_dropout: float = 0.,
                 n_outputs: int = 1,
                 ):
        super().__init__()
        self.input_size = input_size
        self.n_outputs = n_outputs

        if isinstance(shallow_units, list):
            assert len(shallow_units) == input_size
        elif isinstance(shallow_units, int):
            shallow_units = [shallow_units for _ in range(input_size)]

        self.feature_nns = torch.nn.ModuleList([
            FeatureNN(shallow_units=shallow_units[i],
                      hidden_units=hidden_units,
                      shallow_layer=shallow_layer,
                      hidden_layer=hidden_layer,
                      dropout=hidden_dropout,
                      n_outputs=n_outputs)
            for i in range(input_size)
        ])
        self.feature_dropout = torch.nn.Dropout(p=feature_dropout)
        self.bias = torch.nn.Parameter(torch.zeros(self.n_outputs))

    def forward(self, x):
        x_numeric = x[:, :self.input_size]
        x_categorical = x[:, self.input_size:]
        f_out = torch.cat(self._feature_nns(x_numeric), dim=-1)  # [batch, numeric_features*n_outputs]
        f_out = torch.cat([f_out, x_categorical], dim=-1)  # [batch, n_features*n_outputs], n_features includes both numeric and categorical features
        f_out = f_out.reshape((f_out.shape[0], -1, self.n_outputs))  # [batch, n_features, n_outputs]

        # Divide by a constant to ensure that the summed output at initialization does not
        # saturate the sigmoid. This constant could be tuned. Theoretically should it be sqrt(n_features)?
        self.divide_by = f_out.shape[1]  # n_features
        f_out = f_out / self.divide_by
        f_out = self.feature_dropout(f_out)
        self.f_out = f_out
        return f_out.sum(axis=1) + self.bias

    def _feature_nns(self, x):
        """
        For each feature, returns a [batch, n_outputs] tensor.
        """
        return [self.feature_nns[i](x[:, i]) for i in range(self.input_size)]
