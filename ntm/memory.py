"""Neural Turing Machine Memory Module.

This module contains the core imports and components for the Neural Turing Machine's
memory system. It provides the necessary PyTorch components for implementing the
external memory mechanism used in the NTM architecture.

The module imports:
- torch: The main PyTorch library for building neural networks.
- nn: The PyTorch module for defining neural network layers and modules.
- Parameter: A PyTorch class for managing learnable parameters in the model.

The module defines a single class:
- Memory: A PyTorch module that implements the external memory mechanism of the NTM.

"""

import pdb  # ADD THIS LINE

import torch

# from jaxtyping import Float, Tensor
from torch import Tensor, nn
from torch.nn import Parameter


class Memory(nn.Module):
    """Neural Turing Machine's external memory module.

    This module implements the external memory component of a Neural Turing Machine (NTM).
    It provides functionality for reading from and writing to memory, as well as
    initializing and resetting the memory state.

    Attributes:
        _memory_size: Tuple specifying the dimensions of the memory matrix (N, M).
        initial_state: Buffer containing the initial memory state.
        initial_read: Learnable parameter for the initial read vector.
        memory: Current state of the memory matrix.
        _initial_memory: Initial state of the memory matrix, used for resetting.
        M: Current state of the memory matrix.

    """

    def __init__(self, N: int, W: int) -> None:
        """Initialize the NTM memory matrix.

        Args:
            N: Number of memory locations
            W: Size of each memory location
        """
        super().__init__()
        self.N = N
        self.W = W
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize memory with random values (batch_size=1)
        self.register_buffer("_initial_memory", torch.randn(1, self.N, self.W) * 0.05)

    def get_size(self) -> tuple[int, int]:
        return self.N, self.W

    def read(self) -> Tensor:
        return self.M

    def write(self, w: Tensor, e: Tensor, a: Tensor) -> None:
        # Update memory using write weights, erase vector, and add vector
        self.M = self.M * (1 - w.T @ e) + w.T @ a

    def reset(self, batch_size: int) -> None:
        """Initialize memory state."""
        # print(
        #     f"Memory.reset() called with batch_size: {batch_size}"
        # )  # Debug print - Keep this line
        # pdb.set_trace()  # REMOVE or COMMENT OUT THIS LINE - BREAKPOINT
        self.M = self._initial_memory.clone().repeat(batch_size, 1, 1)
        # self.read_weights = self.initial_read_weights.repeat(batch_size, 1) # REMOVE THIS LINE - not used - belongs to heads or NTM state
        # self.write_weights = self.initial_write_weights.repeat(batch_size, 1) # REMOVE THIS LINE - not used - belongs to heads or NTM state

    def get_initial_read(self, batch_size: int) -> torch.Tensor:
        """Get the initial read vector for the memory."""
        # memory_matrix is (1, N, W), repeat to (batch_size, N, W)
        return self.memory_matrix.clone().repeat(batch_size, 1, 1)

    def size(self) -> tuple[int, int]:
        """Get the size of the memory matrix.

        Returns:
            A tuple containing the dimensions of the memory matrix as (N, M), where:
                - N: Number of memory locations
                - M: Vector size at each memory location

        """
        return self.N, self.W
