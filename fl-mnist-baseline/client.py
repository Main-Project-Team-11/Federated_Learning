import flwr as fl
import torch
from collections import OrderedDict

from model import Net
from dataset import load_data
from engine import train, test

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 3
LOCAL_EPOCHS = 1  # small number of local epochs per round, as required

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, partition_id):
        self.partition_id = partition_id
        self.net = Net().to(DEVICE)
        self.train_loader, self.val_loader, _ = load_data(
            partition_id, NUM_CLIENTS, include_test=False
        )

    def get_parameters(self, config):
        """Extract current model weights as a list of NumPy arrays (Flower's format)."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Load weights received from the server into this client's local model."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Called by the server each round:
        1. Load the global weights the server sent.
        2. Train locally for LOCAL_EPOCHS epochs on this client's private data.
        3. Return updated weights + number of samples trained on (used for weighting in FedAvg).
        """
        self.set_parameters(parameters)
        train(self.net, self.train_loader, epochs=LOCAL_EPOCHS, device=DEVICE)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the given (global) weights on this client's local validation data."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.val_loader, device=DEVICE)
        print(f"[Client {self.partition_id}] local val_loss={loss:.4f}, val_acc={accuracy:.4f}")
        return loss, len(self.val_loader.dataset), {"accuracy": accuracy}

def client_fn(cid: str) -> fl.client.Client:
    """Factory function Flower calls to create a client instance for a given client ID (0,1,2)."""
    return FlowerClient(partition_id=int(cid)).to_client()
