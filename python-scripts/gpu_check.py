import torch
import sys

print(f"Python: {sys.version}", flush=True)
print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}", flush=True)
    print(f"Device count: {torch.cuda.device_count()}", flush=True)
    print(f"Device name: {torch.cuda.get_device_name(0)}", flush=True)
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x
    print(f"Matmul test passed, result shape: {y.shape}", flush=True)
else:
    print("ERROR: CUDA not available — will not be able to train CNN on GPU.", flush=True)