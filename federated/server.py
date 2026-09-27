"""
federated/server.py
FedMed Central Aggregation Server — Member 3 Deliverable.

Implements Federated Averaging (FedAvg): weighted parameter aggregation across
participating hospital client state dicts.
"""
import copy
from typing import List, Dict
import torch


def federated_averaging(
    client_weights_list: List[Dict[str, torch.Tensor]],
    client_sample_counts: List[int],
) -> Dict[str, torch.Tensor]:
    """
    Computes FedAvg: sample-weighted average of model state dicts.

    Math:
        w_{global} = sum_{k=1}^K (n_k / N) * w_k
    where N = sum(n_k).

    Args:
        client_weights_list: List of state_dict dictionaries from K clients.
        client_sample_counts: List of integer sample counts n_k from K clients.

    Returns:
        Aggregated global model state_dict.
    """
    total_samples = sum(client_sample_counts)
    assert total_samples > 0, "Total sample count across clients must be positive."
    assert len(client_weights_list) == len(client_sample_counts), (
        f"Length mismatch: {len(client_weights_list)} weight dicts vs "
        f"{len(client_sample_counts)} sample counts."
    )

    # Initialize global weights matching structure of first client's state_dict
    global_weights = copy.deepcopy(client_weights_list[0])

    # Zero out floating point tensors for accumulation
    for key in global_weights.keys():
        if global_weights[key].is_floating_point():
            global_weights[key] = torch.zeros_like(global_weights[key], dtype=torch.float32)

    # Accumulate weighted parameters
    for w_k, n_k in zip(client_weights_list, client_sample_counts):
        weight_factor = n_k / total_samples
        for key in global_weights.keys():
            if global_weights[key].is_floating_point():
                global_weights[key] += w_k[key].to(torch.float32) * weight_factor

    return global_weights
