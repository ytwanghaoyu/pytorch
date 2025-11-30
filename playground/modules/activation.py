import numpy as np
import torch


def softmax_np(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def softmax(x: torch.Tensor) -> torch.Tensor:
    exp = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
    return exp / torch.sum(exp, dim=-1, keepdim=True)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))


if __name__ == "__main__":
    example_np = np.random.rand(2, 4)
    example_np = np.concat([example_np, np.array([[2.0, 1000000.0, 2.0, 2.0]])], axis=0)
    example_np2 = np.random.rand(2, 2, 6, 4)
    example_torch = torch.rand(2, 4)
    example_torch = torch.cat(
        [example_torch, torch.tensor([[2.0, 1000000.0, 2.0, 2.0]])], dim=0
    )

    print(softmax_np(example_np2))
    print(softmax_np(example_np))
    print(softmax(example_torch))
    print(sigmoid(torch.tensor([-100000.0, 0.0, 100000.0])))
