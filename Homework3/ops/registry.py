"""Registry for auto-selecting CUDA vs PyTorch backend."""
try:
    import cuda_ops
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
