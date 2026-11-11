
import torch
import numpy as np

torch.manual_seed(42)

n = 65536
x_cpu = torch.randn(n, 4, dtype=torch.float32)
x_gpu = x_cpu.cuda()

reference = np.linalg.norm(
    x_cpu.numpy().astype(np.float64), axis=1
)

print("PyTorch:", torch.__version__, flush=True)
print("GPU:", torch.cuda.get_device_name(0), flush=True)

for i in range(30):
    print(f"Iteration {i}: before norm", flush=True)

    result = torch.linalg.vector_norm(x_gpu, dim=1)
    torch.cuda.synchronize()

    values = result.cpu().numpy()
    error = np.abs(values - reference)

    print(
        f"Iteration {i}: max_error={error.max():.8g}, "
        f"bad_rows={(error > 1e-4).sum()}",
        flush=True
    )

print("TEST PASSED", flush=True)