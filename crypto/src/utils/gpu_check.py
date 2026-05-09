# crypto/src/utils/gpu_check.py
import torch
from loguru import logger


def verify_gpu(verbose: bool = True) -> dict:
    """
    Verifies CUDA / GPU availability and returns a summary dict.
    Raises RuntimeError if CUDA is not available.
    """
    result = {
        "cuda_available": torch.cuda.is_available(),
        "device_count":   torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name":    None,
        "vram_total_gb":  None,
        "vram_free_gb":   None,
        "cuda_version":   torch.version.cuda,
        "torch_version":  torch.__version__,
    }

    if not result["cuda_available"]:
        logger.error("CUDA is NOT available. Check your PyTorch CUDA build and NVIDIA drivers.")
        raise RuntimeError(
            "CUDA not available.\n"
            "Fix: pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 "
            "--index-url https://download.pytorch.org/whl/cu121"
        )

    props = torch.cuda.get_device_properties(0)
    result["device_name"]   = props.name
    result["vram_total_gb"] = round(props.total_memory / 1e9, 2)

    free_mem, _            = torch.cuda.mem_get_info(0)
    result["vram_free_gb"] = round(free_mem / 1e9, 2)

    if verbose:
        separator = "─" * 45
        logger.info(separator)
        logger.info("  GPU CHECK")
        logger.info(separator)
        logger.info(f"  CUDA Available  : {result['cuda_available']}")
        logger.info(f"  Device          : {result['device_name']}")
        logger.info(f"  VRAM Total      : {result['vram_total_gb']} GB")
        logger.info(f"  VRAM Free       : {result['vram_free_gb']} GB")
        logger.info(f"  CUDA Version    : {result['cuda_version']}")
        logger.info(f"  PyTorch Version : {result['torch_version']}")
        logger.info(separator)

    return result


if __name__ == "__main__":
    verify_gpu()