import torch


def get_cuda_devices():
    result = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            result.append(f"CUDA {i}: " + torch.cuda.get_device_name(i))
    return result
