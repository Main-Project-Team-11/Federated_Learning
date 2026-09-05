# Repository Guidelines

## Project Structure & Module Organization

The application lives in `fl-mnist-baseline/`:

- `sim.py` starts the three-client Flower simulation and configures FedAvg.
- `client.py` implements the Flower client lifecycle.
- `dataset.py` loads MNIST and creates deterministic client partitions.
- `model.py` defines the shared PyTorch CNN.
- `engine.py` contains local `train` and `test` loops.
- `data/MNIST/raw/` is a downloaded dataset cache; do not edit or commit generated copies.

There is currently no `tests/` directory, package manifest, or README. Add new tests under `fl-mnist-baseline/tests/` and keep reusable functionality in the existing module matching its concern.

## Development Commands

Run the simulation from the application directory:

```powershell
cd fl-mnist-baseline
python sim.py
```

This starts three synchronous federated rounds using FedAvg. The checked-in virtual environments do not include runtime dependencies. Use a dedicated environment and install compatible `torch`, `torchvision`, `flwr`, and Flower's simulation dependencies before running. When dependencies are formalized, add a lockfile or requirements file and document its install command.

## Coding Style & Naming Conventions

Use Python with four-space indentation and PEP 8 conventions. Use `snake_case` for functions, variables, and modules; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants such as `NUM_CLIENTS`. Add concise docstrings to public functions and classes. Keep device placement explicit (`tensor.to(DEVICE)`) and preserve the NumPy-array boundary required by Flower parameter exchange.

No formatter, linter, or type checker is configured. Do not introduce broad formatting-only changes with functional work.

## Testing Guidelines

No automated tests exist yet. Add `pytest` tests named `test_*.py`, especially for data partition sizes, model output shape, parameter round trips, and loss/accuracy calculations. Avoid making tests download MNIST; use small synthetic `TensorDataset` fixtures. Run tests from the application directory:

```powershell
python -m pytest tests
```

## Commits & Pull Requests

Git history contains no commits, so no established message convention can be inferred. Use short imperative messages, for example `Add client partition tests`. Keep each commit focused. Pull requests should describe the behavioral change, list commands run, identify dependency or data changes, and include observed metrics when altering training or aggregation behavior.

## Data and Reproducibility

MNIST is downloaded automatically by TorchVision. Never commit virtual environments, `__pycache__/`, or downloaded data. Seed any new randomness and state whether a change affects reproducibility, client partitioning, or the evaluation split.
