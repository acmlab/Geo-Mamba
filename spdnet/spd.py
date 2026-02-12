import torch
from torch import nn
from spdnet import StiefelParameter
import numpy as np


def is_symmetric(matrix):
    return torch.allclose(matrix, matrix.transpose(-1, -2))

def is_positive_definite(matrix):
    eigvals, _ = torch.linalg.eigh(matrix)
    return torch.all(eigvals > 0)

def is_spd(batch_matrix):
    for i in range(batch_matrix.shape[0]):
        matrix = batch_matrix[i]
        if not is_symmetric(matrix):
            print(f"Matrix {i} is not symmetric.")
            return False
        if not is_positive_definite(matrix):
            print(f"Matrix {i} is not positive definite.")
            return False
    return True

class SPDTransform(nn.Module):
    def __init__(self, input_size, output_size, in_channels=1):

        super(SPDTransform, self).__init__()

        if in_channels > 1:
            self.weight = StiefelParameter(
                torch.Tensor(in_channels, input_size, output_size), requires_grad=True
            )
        else:
            self.weight = StiefelParameter(
                torch.Tensor(input_size, output_size), requires_grad=True
            )

        nn.init.orthogonal_(self.weight)

    def forward(self, input):

        weight = self.weight
        output = weight.transpose(-2, -1) @ input @ weight
        output = 0.5 * (output + output.transpose(-2, -1))
        return output

class SPDTangentSpace(nn.Module):
    def __init__(self):

        super(SPDTangentSpace, self).__init__()

    def forward(self, input):

        s, u = torch.linalg.eigh(input)
        s = s.clamp(min=1e-4, max=1e6)
        s = s.log().diag_embed()
        output = u @ s @ u.transpose(-2, -1)
        return torch.flatten(output, 1)

class SPDVectorize(nn.Module):

    def __init__(self, vectorize_all=True):
        super(SPDVectorize, self).__init__()
        self.register_buffer('vectorize_all', torch.tensor(vectorize_all))

    def forward(self, input):
        row_idx, col_idx = np.triu_indices(input.shape[-1])
        output = input[..., row_idx, col_idx]

        if self.vectorize_all:
            output = torch.flatten(output, 1)
        return output

class SPDTangentSpace1(nn.Module):

    def __init__(self, vectorize=True, vectorize_all=True):
        super(SPDTangentSpace1, self).__init__()
        self.vectorize = vectorize
        if vectorize:
            self.vec = SPDVectorize(vectorize_all=vectorize_all)

    def forward(self, input):

        u, s, v = torch.svd(input)
        s = s.clamp(min=1e-4, max=1e4)
        s = s.log().diag_embed()
        if s.isnan().any():
            print('SPDTangentSpace log negative')
            raise ValueError
        output = u @ s @ u.transpose(-2, -1)
        if self.vectorize:
            output = self.vec(output if len(output.shape) > 2 else output.unsqueeze(0))

        return output

class SPDExpMap(nn.Module):
    def __init__(self):

        super(SPDExpMap, self).__init__()

    def forward(self, input):
        s, u = torch.linalg.eigh(input)
        s = s.exp().diag_embed()
        output = u @ s @ u.transpose(-2, -1)
        return output

class SPDRectified(nn.Module):
    def __init__(self, epsilon=1e-4):
        super(SPDRectified, self).__init__()
        self.register_buffer('epsilon', torch.DoubleTensor([epsilon]))

    def forward(self, input):
        s, u = torch.linalg.eigh(input)
        s = s.clamp(min=self.epsilon[0])
        s = s.diag_embed()
        output = u @ s @ u.transpose(-2, -1)

        return output
