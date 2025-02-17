from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ntm.controller import Controller
from ntm.head import ReadHead, WriteHead
from ntm.memory import Memory

ControllerState: TypeAlias = tuple[Tensor, ...]
# One more more tensors followed by a ControllerState
StateTuple: TypeAlias = tuple[Tensor, *tuple[Tensor, ...], ControllerState]


class NTM(nn.Module):
    def __init__(
        self,
        vector_length: int,
        hidden_size: int,
        memory_size: tuple[int, int],  # (N, W)
        controller: Controller,  # Now takes a Controller instance
    ) -> None:
        """Initialize the Neural Turing Machine.

        Args:
            vector_length: Length of input/output vectors.
            hidden_size: Size of hidden layers.
            memory_size: Tuple of (memory locations, memory vector size).
            controller: An instance of a Controller implementation.
        """
        super().__init__()
        self.controller = controller
        self.memory = Memory(memory_size[0], memory_size[1])
        self.read_head = ReadHead(self.memory, hidden_size)
        self.write_head = WriteHead(self.memory, hidden_size)
        self.fc = nn.Linear(hidden_size + memory_size[1], vector_length)
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)

    def get_initial_state(self, batch_size: int = 1) -> StateTuple:
        """Get the initial state of the NTM.

        Args:
            batch_size: Size of the batch.

        Returns:
            Tuple containing initial states for all components.
        """
        self.memory.reset(batch_size)
        controller_state = self.controller.get_initial_state(batch_size)
        # Initialize initial read vector to zeros, shape [batch_size, W]
        read = torch.zeros(batch_size, self.memory.W)
        read_head_state = self.read_head.get_initial_state(batch_size)
        write_head_state = self.write_head.get_initial_state(batch_size)
        return (read, read_head_state, write_head_state, controller_state)

    def forward(
        self,
        x: Tensor,
        previous_state: StateTuple,
    ) -> tuple[Tensor, StateTuple]:
        """Process input through the NTM.

        Args:
            x: Input tensor of shape [batch_size, vector_length + 1]
            previous_state: Tuple containing previous controller state and head weights

        Returns:
            Tuple containing:
                - Output tensor
                - New state tuple
        """
        # print("ENTERING NTM.FORWARD")
        previous_read, previous_read_weights, previous_write_weights, previous_controller_state = (
            previous_state
        )

        # print(f"Type of previous_read before read_head: {type(previous_read)}")
        # print(f"Shape of previous_read before read_head: {previous_read.shape}")

        # Prepare controller input: concatenate input with previous read *vector*
        controller_input: Tensor = torch.cat([x, previous_read], dim=1)

        # print("controller_input", flush=True)
        # Process through controller
        controller_output, controller_state = self.controller(
            controller_input, previous_controller_state
        )
        # print("CONTROLLER OUTPUT SHAPE:", controller_output.shape)
        # print(f"Shape of previous_read before read_head: {previous_read.shape}")
        read_vector, read_weights = self.read_head(controller_output, previous_read_weights)

        # Write to memory (input to write_head is controller_output)
        write_weights = self.write_head(controller_output, previous_write_weights)

        # Update memory state (no explicit update needed, memory is updated in heads)

        # Concatenate controller output and read vector for final output layer
        fc_input = torch.cat([controller_output, read_vector], dim=1)
        output_vector = self.fc(fc_input)
        output_vector = torch.sigmoid(output_vector)

        # The new read for the next step is the current read_vector
        return output_vector, (read_vector, read_weights, write_weights, controller_state)
