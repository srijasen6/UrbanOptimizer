import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Basic tensor operation:", torch.rand(2,3) @ torch.rand(3,2))