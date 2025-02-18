from abc import ABC, abstractmethod
from typing import Literal, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

# NOT SURE

type StateTuple = tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor]]
type FFState = tuple[Literal[0], Literal[0]]


class ControllerProtocol(Protocol):
    """Protocol defining the interface for all controller implementations."""

    def forward(self, x: Tensor, state: tuple) -> tuple[Tensor, tuple]:
        """Process input and state through the controller.

        Args:
            x: Input tensor.
            state: Current controller state.

        Returns:
            Tuple of output tensor and new state.
        """
        ...

    def get_initial_state(self, batch_size: int) -> tuple:
        """Get initial state for the controller.

        Args:
            batch_size: Size of the batch.

        Returns:
            Initial state tuple.
        """
        ...


class Controller(nn.Module, ABC):
    """Abstract base class for Neural Turing Machine controllers.

    The Controller is a core component of the NTM architecture that processes inputs and
    generates outputs while maintaining internal state. It serves as the interface between
    the external environment and the NTM's memory system.

    Attributes:
        vector_length: Length of input vectors.
        hidden_size: Size of hidden layers in the controller.

    Subclasses must implement:
        - forward(): Process input through the controller
        - get_initial_state(): Initialize controller state

    The class provides common functionality for:
        - Parameter initialization using Xavier-like initialization
        - Maintaining consistent interface through the ControllerProtocol

    Note:
        This is an abstract base class that should be subclassed to implement specific
        controller architectures (e.g., LSTM, FeedForward).
    """

    def __init__(self, vector_length: int, hidden_size: int) -> None:
        """Initialize the controller.

        Args:
            vector_length: Length of input vectors.
            hidden_size: Size of hidden layers.
        """
        super().__init__()
        self.vector_length = vector_length
        self.hidden_size = hidden_size

    @abstractmethod
    def forward(self, x: Tensor, state: tuple) -> tuple[Tensor, tuple]:
        """Process input through the controller.

        Args:
            x: Input tensor of shape [batch_size, vector_length].
            state: Current controller state tuple.

        Returns:
            Tuple containing:
                - Output tensor of shape [batch_size, hidden_size]
                - New state tuple
        """
        pass

    @abstractmethod
    def get_initial_state(self, batch_size: int) -> tuple:
        """Get initial state for the controller.

        Args:
            batch_size: Number of sequences in the batch.

        Returns:
            Initial state tuple appropriate for the specific controller implementation.
        """
        pass

    def _initialize_parameters(self) -> None:
        """Initialize the parameters of the controller using Xavier-like initialization.

        For LSTM:
            - Biases are initialized to 0.
            - Weights are uniformly initialized with a scale of 5 / sqrt(input_size + hidden_size).
        For Feedforward:
            - Weights are uniformly initialized with the same scale.
        """
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)  # Initialize biases to 0
            else:
                stdev = 5 / (np.sqrt(self.vector_length + self.hidden_size))
                nn.init.uniform_(param, -stdev, stdev)  # Initialize weights uniformly


class LSTMController(Controller):
    """LSTM-based controller implementation."""

    def __init__(self, vector_length: int, hidden_size: int):
        """Initialize the LSTM controller.

        Args:
            vector_length: Length of input vectors.
            hidden_size: Size of hidden layers.
            memory_vector_length: Width of memory vectors. Defaults to vector_length if not specified.
        """
        super().__init__(vector_length, hidden_size)
        # Use memory_vector_length if specified, otherwise use vector_length
        # self.memory_vector_length = memory_vector_length or vector_length

        # Input consists of:
        # - Original input (vector_length)
        # - Memory read (memory_vector_length)
        # total_input_size = vector_length + self.memory_vector_length
        self.layer = nn.LSTM(input_size=vector_length, hidden_size=hidden_size)
        self.lstm_h_state = Parameter(torch.randn(1, 1, hidden_size) * 0.05)
        self.lstm_c_state = Parameter(torch.randn(1, 1, hidden_size) * 0.05)
        self._initialize_parameters()
        print(f"init LSTM, {vector_length=}, {hidden_size=}")

        # TEMPORARY DEFINITIONS to avoid AttributeError for initial run
        # self.layer_1 = nn.Linear(1, 1)  # Dummy linear layer
        # self.layer_2 = nn.Linear(1, 1)  # Dummy linear layer
        # self.layer_3 = nn.Linear(1, 1)  # Dummy linear layer
        # These layers are just defined to prevent AttributeError, their outputs are not used meaningfully in this test.

    def forward(
        self, x: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Process input through the LSTM controller.

        Args:
            x: Input tensor.
            state: Current LSTM state (hidden, cell).

        Returns:
            Tuple of output tensor and new state.
        """
        print(f"LSTMController.forward input x shape: {x.shape}")  # 4, 29
        output, state = self.layer(x.unsqueeze(0), state)
        print(f"LSTM Output shape: {output.shape}")
        print("LSTM Output (first batch element):\n", output[0].detach().cpu().numpy())
        return output.squeeze(0), state

    def get_initial_state(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Get initial state for the LSTM controller.

        Args:
            batch_size: Size of the batch.

        Returns:
            Tuple of initial hidden and cell states.
        """
        lstm_h = self.lstm_h_state.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_state.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c


class FeedForwardController(Controller):
    """Feed-forward controller implementation."""

    def __init__(self, vector_length: int, hidden_size: int) -> None:
        """Initialize the feed-forward controller.

        Args:
            vector_length: Length of input vectors.
            hidden_size: Size of hidden layers.
            memory_vector_size: Size of memory vectors.
        """
        super().__init__(vector_length, hidden_size)
        self.layer_1 = nn.Linear(vector_length, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self._initialize_parameters()

    def forward(self, x: Tensor, previous_state: None) -> tuple[Tensor, None]:
        """Process input through the feedforward controller."""
        # print(f"FeedForwardController.forward input shape: {x.shape}")
        # print(f"FeedForwardController.layer_1 weight shape: {self.layer_1.weight.shape}")
        x1 = F.relu(self.layer_1(x))
        output = self.layer_2(x1)
        return output, None

    def get_initial_state(self, batch_size: int) -> None:
        """Get initial state for feedforward controller (None)."""
        return None
