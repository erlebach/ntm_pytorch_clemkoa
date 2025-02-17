import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from ntm.memory import Memory
from ntm.utils import _convolve


class Head(nn.Module):
    def __init__(self, memory: Memory, hidden_size: int) -> None:
        super(Head, self).__init__()
        self.memory = memory
        memory_length, memory_vector_length = memory.get_size()
        # (k : vector, beta: scalar, g: scalar, s: vector, gamma: scalar)
        self.k_layer = nn.Linear(hidden_size, memory_vector_length)
        self.beta_layer = nn.Linear(hidden_size, 1)
        self.g_layer = nn.Linear(hidden_size, 1)
        self.s_layer = nn.Linear(hidden_size, 3)
        self.gamma_layer = nn.Linear(hidden_size, 1)
        for layer in [self.k_layer, self.beta_layer, self.g_layer, self.s_layer, self.gamma_layer]:
            nn.init.xavier_uniform_(layer.weight, gain=1.4)
            nn.init.normal_(layer.bias, std=0.01)

        self._initial_state = Parameter(torch.randn(1, self.memory.get_size()[0]) * 1e-5)

    def get_initial_state(self, batch_size: int) -> Tensor:
        # Softmax to ensure weights are normalized
        return F.softmax(self._initial_state, dim=1).repeat(batch_size, 1)

    def get_head_weight(self, x: Tensor, previous_state: Tensor, memory_read: Tensor) -> Tensor:
        """Get the head weight."""
        # print(f"Shape of x in get_head_weight: {x.shape}")  # Debug print
        k: Tensor = self.k_layer(x)
        beta = F.softplus(self.beta_layer(x))
        g = F.sigmoid(self.g_layer(x))
        s = F.softmax(self.s_layer(x), dim=1)
        gamma = 1 + F.softplus(self.gamma_layer(x))

        # print(f"Shape of memory_read before cosine_similarity: {memory_read.shape}")  # Debug print
        # print(
        #     f"Shape of k.unsqueeze(1) before cosine_similarity: {k.unsqueeze(1).shape}"
        # )  # Debug print

        # Calculate content-based addressing
        wc = F.softmax(
            beta * F.cosine_similarity(k.unsqueeze(1) + 1e-16, memory_read + 1e-16, dim=-1), dim=1
        )
        # print(f"Shape of wc after cosine_similarity: {wc.shape}")  # ADDED: Debug print
        # Focusing by location
        w_g = g * wc + (1 - g) * previous_state
        w_t = self.shift(w_g, s)
        w = w_t**gamma
        w = torch.div(w, torch.sum(w, dim=1).unsqueeze(1) + 1e-16)
        # print(f"Shape of w before return: {w.shape}")  # ADDED: Debug print
        return w

    def shift(self, w_g: Tensor, s: Tensor) -> Tensor:
        result = w_g.clone()
        for b in range(len(w_g)):
            result[b] = _convolve(w_g[b], s[b])
        return result


class ReadHead(Head):
    def forward(self, x, previous_state) -> tuple[Tensor, Tensor]:
        memory_read = self.memory.read()
        w = self.get_head_weight(x, previous_state, memory_read)
        return torch.matmul(w.unsqueeze(1), memory_read).squeeze(1), w


class WriteHead(Head):
    def __init__(self, memory, hidden_size):
        super(WriteHead, self).__init__(memory, hidden_size)
        memory_length, memory_vector_length = memory.get_size()
        self.e_layer = nn.Linear(hidden_size, memory_vector_length)
        self.a_layer = nn.Linear(hidden_size, memory_vector_length)
        for layer in [self.e_layer, self.a_layer]:
            nn.init.xavier_uniform_(layer.weight, gain=1.4)
            nn.init.normal_(layer.bias, std=0.01)

    def forward(self, x, previous_state) -> Tensor:
        memory_read = self.memory.read()
        w = self.get_head_weight(x, previous_state, memory_read)
        e = F.sigmoid(self.e_layer(x))
        a: Tensor = self.a_layer(x)

        # write to memory (w, memory, e , a)
        self.memory.write(w, e, a)
        return w
