import torch

# 1. 检查版本（应该包含 git hash，例如 2.6.0a0+git...）
print(f"PyTorch Version: {torch.__version__}")

# 2. 检查 MPS (Metal) 是否可用
print(f"CUDA Available: {torch.cuda.is_available()}")
print("Checking torch.backends:")
for attr in dir(torch.backends):
    if not attr.startswith("_"):
        module = getattr(torch.backends, attr)
        if hasattr(module, "is_available"):
            print(f"  {attr}: {module.is_available()}")

# 3. 跑一个简单的张量计算测试
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.randn(3, 3, device=device)
    y = torch.randn(3, 3, device=device)
    print("Result on M4 GPU:\n", x + y)
else:
    print("MPS not detected!")
