# crypto/src/utils/gpu_check.py
"""
GPU / Hardware Verification Utility.

Checks CUDA availability gracefully. Falls back to CPU mode rather than
terminating execution if CUDA is unavailable.
"""

from loguru import logger


def verify_gpu(verbose: bool = True, raise_on_fail: bool = False) -> dict:
    """
    Verifies CUDA / GPU availability and returns a summary dict.
    Does NOT crash on CPU systems unless raise_on_fail is True.
    """
    try:
        import torch
        torch_installed = True
        cuda_avail = torch.cuda.is_available()
    except ImportError:
        torch = None
        torch_installed = False
        cuda_avail = False

    result = {
        "torch_installed": torch_installed,
        "cuda_available":  cuda_avail,
        "device_count":    torch.cuda.device_count() if cuda_avail and torch else 0,
        "device_name":     None,
        "vram_total_gb":   None,
        "vram_free_gb":    None,
        "cuda_version":    torch.version.cuda if torch and hasattr(torch, "version") else None,
        "torch_version":   torch.__version__ if torch else None,
    }

    if not cuda_avail:
        logger.warning(
            "CUDA hardware acceleration is NOT active. Falling back to local CPU execution.\n"
            "Tip: For GPU acceleration, install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        )
        if raise_on_fail:
            raise RuntimeError("CUDA is required but not available.")
        return result

    try:
        props = torch.cuda.get_device_properties(0)
        result["device_name"]   = props.name
        result["vram_total_gb"] = round(props.total_memory / 1e9, 2)

        free_mem, _            = torch.cuda.mem_get_info(0)
        result["vram_free_gb"] = round(free_mem / 1e9, 2)
    except Exception as e:
        logger.warning(f"Could not query CUDA device properties: {e}")

    if verbose:
        separator = "─" * 45
        logger.info(separator)
        logger.info("  HARDWARE CHECK")
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
