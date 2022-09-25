import os
from torch.cuda import device_count

# manage GPUs
default_gpuid = "MIG-198d257f-725f-529c-ac47-ffc6f4fdf544"


def gpu_manager(gpuid: str = default_gpuid):
    """
    Set GPU by passing GPU-ID. GPU-ID can be obtained with "nvidia-smi -L"
    in terminal. "nvidia-smi" returns current GPU usage.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    print('Number of Devices: ', device_count())