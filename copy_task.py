import argparse
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, optim
from torch.utils.tensorboard.writer import SummaryWriter

from ntm.controller import FeedForwardController, LSTMController
from ntm.ntm import NTM
from ntm.utils import plot_copy_results

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--train", help="Trains the model", action="store_true")
parser.add_argument("--ff", help="Feed forward controller", action="store_true")
parser.add_argument(
    "--eval",
    help="Evaluates the model. Default path is models/copy.pt",
    action="store_true",
)
parser.add_argument(
    "--modelpath",
    help="Specify the model path to load, for training or evaluation",
    type=str,
)
parser.add_argument(
    "--epochs",
    help="Specify the number of epochs for training",
    type=int,
    default=50_000,
)
args = parser.parse_args()

seed = 1
random.seed(seed)

np.random.seed(seed)
# rng = np.random.default_rng(seed)

torch.manual_seed(seed)


def get_training_sequence(
    sequence_min_length: int,
    sequence_max_length: int,
    vector_length: int,
    batch_size: int = 1,
) -> tuple[Tensor, Tensor]:
    """Generate a training sequence for the copy task.

    Args:
        sequence_min_length: Minimum length of the sequence.
        sequence_max_length: Maximum length of the sequence.
        vector_length: Length of each vector in the sequence.
        batch_size: Number of sequences to generate in parallel.

    Returns:
        A tuple containing:
            - input_seq: The input sequence with an end marker (shape: [sequence_length+1, batch_size, vector_length+1])
            - output: The target output sequence (shape: [sequence_length, batch_size, vector_length])

    """
    #sequence_length = rng.integers(
        #sequence_min_length,
        #sequence_max_length,
        #endpoint=True,
    #)
    sequence_length = random.randint(sequence_min_length, sequence_max_length)

    output = torch.bernoulli(
        torch.Tensor(sequence_length, batch_size, vector_length).uniform_(
            0,
            1,
        ),
    )
    input_seq = torch.zeros(sequence_length + 1, batch_size, vector_length + 1)
    input_seq[:sequence_length, :, :vector_length] = output
    input_seq[sequence_length, :, vector_length] = 1.0
    return input_seq, output


def train(epochs: int = 50_000) -> None:
    """Train the Neural Turing Machine (NTM) model on the copy task.

    The function handles the entire training process including:
    - Setting up TensorBoard logging
    - Initializing model parameters
    - Creating and configuring the NTM model
    - Running the training loop
    - Calculating and logging loss and cost metrics
    - Saving the trained model

    Args:
        epochs: Number of training epochs to run. Defaults to 50,000.

    Returns:
        None. The trained model is saved to disk at the specified path.

    """
    # Eastern Standard Time
    # tzz = timezone(timedelta(hours=-5)).strftime("%Y-%m-%dT%H%M%S")
    tensorboard_log_folder = f"runs/copy-task-{datetime.now()}"  # .astimezone(tzz)}"
    writer = SummaryWriter(tensorboard_log_folder)
    print(f"Training for {epochs} epochs, logging in {tensorboard_log_folder}")
    sequence_min_length = 1
    sequence_max_length = 20
    vector_length = 8
    memory_size = (128, 20)  # N, W (#locations x width per loc)
    hidden_layer_size = 100
    batch_size = 4
    lstm_controller = not args.ff
    # memory_locations = memory_size[0]
    # memory_vector_size = memory_size[1]

    writer.add_scalar("sequence_min_length", sequence_min_length)
    writer.add_scalar("sequence_max_length", sequence_max_length)
    writer.add_scalar("vector_length", vector_length)
    writer.add_scalar("memory_size0", memory_size[0])
    writer.add_scalar("memory_size1", memory_size[1])
    writer.add_scalar("hidden_layer_size", hidden_layer_size)
    writer.add_scalar("lstm_controller", lstm_controller)
    writer.add_scalar("seed", seed)
    writer.add_scalar("batch_size", batch_size)

    # Initialize the controller based on the lstm_controller flag
    if lstm_controller:
        # print(f"lstm_controller, {vector_length=}, {hidden_layer_size=}")
        controller = LSTMController(
            vector_length + 1 + memory_size[1],
            hidden_layer_size,
        )
    else:
        controller = FeedForwardController(vector_length+1+memory_size[1], hidden_layer_size)

    # Pass the controller instance to NTM
    model = NTM(vector_length, hidden_layer_size, memory_size, controller)

    optimizer = optim.RMSprop(model.parameters(), momentum=0.9, alpha=0.95, lr=1e-4)
    feedback_frequency = 100
    total_loss = []
    total_cost = []

    Path("models").mkdir(exist_ok=True)
    model_path = "models/copy.pt"
    # if Path(model_path).exists():
    #     print(f"Loading model from {model_path}")
    #     checkpoint = torch.load(model_path, map_location=torch.device("mps"))  # cpu
    #     model.load_state_dict(checkpoint)

    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        input_seq, target = get_training_sequence(
            sequence_min_length,
            sequence_max_length,
            vector_length,
            batch_size,
        )
        state = model.get_initial_state(batch_size)
        for vector in input_seq:
            _, state = model(vector, state)
        y_out = torch.zeros(target.size())
        for j in range(len(target)):
            y_out[j], state = model(torch.zeros(batch_size, vector_length + 1), state)
        loss = F.binary_cross_entropy(y_out, target)

        # print("Target (first batch element):\n", target[0].detach().cpu().numpy())
        # print("Output (first batch element):\n", y_out[0].detach().cpu().numpy())
        # print("Loss value:", loss.item())

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
        y_out_binarized = y_out.clone().data
        y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)
        cost = torch.sum(torch.abs(y_out_binarized - target)) / len(target)
        total_cost.append(cost.item())
        if epoch % feedback_frequency == 0:
            running_loss = sum(total_loss) / len(total_loss)
            running_cost = sum(total_cost) / len(total_cost)
            print(f"Batch: {epoch}, Loss: {running_loss:.6f}")
            with open("original_lstm_loss_1000steps.txt", "a") as f:
                f.write(f"Batch: {epoch}, Loss: {running_loss:.6f}\n")
            total_loss = []
            total_cost = []

    # torch.save(model.state_dict(), model_path)
    # Add these lines at the end of the `train` function in copy_task.py, after the training loop:
    model_path_original_lstm = "models/original_lstm_model_1000steps.pt"
    torch.save(model.state_dict(), model_path_original_lstm)
    print(f"Saved original LSTM model weights to: {model_path_original_lstm}")


def eval_model(model_path: str) -> None:
    """Evaluate a trained Neural Turing Machine model on the copy task.

    Loads a pre-trained model from the specified path and evaluates its performance
    on sequences of different lengths. Plots the results of the copy task.

    Args:
        model_path: Path to the saved model checkpoint file.

    Returns:
        None

    """
    vector_length = 8
    memory_size = (128, 20)
    hidden_layer_size = 100
    lstm_controller = not args.ff
    memory_locations = memory_size[0]
    memory_vector_size = memory_size[1]

    model = NTM(vector_length, hidden_layer_size, memory_size, lstm_controller)

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device("mps"))  # cpu
    model.load_state_dict(checkpoint)
    model.eval()

    lengths = [20, 100]
    for el in lengths:
        sequence_length = el
        input_seq, target = get_training_sequence(
            sequence_length,
            sequence_length,
            vector_length,
        )
        state = model.get_initial_state()
        for vector in input_seq:
            _, state = model(vector, state)
        y_out = torch.zeros(target.size())
        for j in range(len(target)):
            y_out[j], state = model(torch.zeros(1, vector_length + 1), state)
        y_out_binarized = y_out.clone().data
        y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

        plot_copy_results(target, y_out, vector_length)


if __name__ == "__main__":
    model_path = "models/copy.pt"
    if args.modelpath:
        model_path = args.modelpath
    if args.train:
        print("train")
        train(args.epochs)
    if args.eval:
        print("eval")
        eval_model(model_path)
