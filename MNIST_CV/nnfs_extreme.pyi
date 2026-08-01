"""Type stubs for nnfs_extreme — compiled C++ extension (pybind11)."""

import numpy as np
from numpy.typing import NDArray

class Network:
    """Neural network loaded from an exported NNFS_Extreme model binary."""

    def __init__(self, model_path: str) -> None:
        """Load a trained network from a binary model file.

        Args:
            model_path: Path to the .bin file produced by Network::export_model().
        """
        ...

    def feedforward(self, input: NDArray[np.float32]) -> list[float]:
        """Run a single forward pass through the network.

        Args:
            input: Flat float32 numpy array of shape (784,) for MNIST.

        Returns:
            List of 10 floats — one activation per output neuron (digit 0-9).
        """
        ...
