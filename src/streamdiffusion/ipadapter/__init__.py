try:
    from .base_ipadapter_pipeline import BaseIPAdapterPipeline
except Exception as e:
    print(f"ipadapter.__init__: Failed to import BaseIPAdapterPipeline: {e}")
    raise

__all__ = [
    "BaseIPAdapterPipeline",
] 