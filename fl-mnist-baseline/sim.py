import flwr as fl
import torch
from collections import OrderedDict

from model import Net
from dataset import load_data
from engine import test
from client import client_fn, NUM_CLIENTS, DEVICE

def get_evaluate_fn():
    """
    Returns a function the SERVER uses to test the aggregated global model
    on a shared, held-out test set (60,000 train images were split among
    clients; the 10,000 official MNIST test images are used here, untouched
    by any client during training).
    """
    _, _, test_loader = load_data(0, NUM_CLIENTS)
    net = Net().to(DEVICE)

    def evaluate(server_round, parameters, config):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        loss, accuracy = test(net, test_loader, device=DEVICE)
        print(f"\n[Server] Round {server_round} - GLOBAL model test_loss={loss:.4f}, test_acc={accuracy:.4f}\n")
        return loss, {"accuracy": accuracy}

    return evaluate

# FedAvg = baseline aggregation strategy (Requirement 9: not final, just our starting point)
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,           # use 100% of available clients each round for training
    fraction_evaluate=1.0,      # use 100% of available clients each round for local evaluation
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_fn=get_evaluate_fn(),  # server-side global evaluation
)

if __name__ == "__main__":
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=3),  # Requirement 10: at least 3 rounds
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )
