import torch
import logging
from pathlib import Path
from typing import Optional
import fire

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.error("TensorRT not available. Please install it first.")

try:
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logger.error("torchvision not available. Please install it first.")


def export_raft_to_onnx(
    onnx_path: Path,
    resolution: int = 512,
    device: str = "cuda"
) -> bool:
    """
    Export RAFT model to ONNX format
    
    Args:
        onnx_path: Path to save the ONNX model
        resolution: Input resolution for the model
        device: Device to use for export
        
    Returns:
        True if successful, False otherwise
    """
    if not TORCHVISION_AVAILABLE:
        logger.error("torchvision is required but not installed")
        return False
    
    logger.info(f"Exporting RAFT model to ONNX: {onnx_path}")
    logger.info(f"Resolution: {resolution}x{resolution}")
    
    try:
        # Load RAFT model
        logger.info("Loading RAFT Small model...")
        raft_model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=True)
        raft_model = raft_model.to(device=device)
        raft_model.eval()
        
        # Create dummy inputs
        dummy_frame1 = torch.randn(1, 3, resolution, resolution).to(device)
        dummy_frame2 = torch.randn(1, 3, resolution, resolution).to(device)
        
        # Apply RAFT preprocessing if available
        weights = Raft_Small_Weights.DEFAULT
        if hasattr(weights, 'transforms') and weights.transforms is not None:
            transforms = weights.transforms()
            dummy_frame1, dummy_frame2 = transforms(dummy_frame1, dummy_frame2)
        
        dynamic_axes = {
            "frame1": {0: "batch_size"},
            "frame2": {0: "batch_size"},
            "flow": {0: "batch_size"},
        }
        
        logger.info("Exporting to ONNX...")
        with torch.no_grad():
            torch.onnx.export(
                raft_model,
                (dummy_frame1, dummy_frame2),
                str(onnx_path),
                verbose=False,
                input_names=['frame1', 'frame2'],
                output_names=['flow'],
                opset_version=17,
                export_params=True,
                dynamic_axes=dynamic_axes,
            )
        
        del raft_model
        torch.cuda.empty_cache()
        
        logger.info(f"Successfully exported ONNX model to {onnx_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export ONNX model: {e}")
        import traceback
        traceback.print_exc()
        return False


def build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    resolution: int = 512,
    fp16: bool = True,
    workspace_size_gb: int = 4
) -> bool:
    """
    Build TensorRT engine from ONNX model
    
    Args:
        onnx_path: Path to the ONNX model
        engine_path: Path to save the TensorRT engine
        resolution: Input resolution for optimization
        fp16: Enable FP16 precision mode
        workspace_size_gb: Maximum workspace size in GB
        
    Returns:
        True if successful, False otherwise
    """
    if not TENSORRT_AVAILABLE:
        logger.error("TensorRT is required but not installed")
        return False
    
    if not onnx_path.exists():
        logger.error(f"ONNX model not found: {onnx_path}")
        return False
    
    logger.info(f"Building TensorRT engine from ONNX model: {onnx_path}")
    logger.info(f"Output path: {engine_path}")
    logger.info(f"Resolution: {resolution}x{resolution}")
    logger.info(f"FP16 mode: {fp16}")
    logger.info("This may take several minutes...")
    
    try:
        builder = trt.Builder(trt.Logger(trt.Logger.INFO))
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
        
        logger.info("Parsing ONNX model...")
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                logger.error("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    logger.error(f"Parser error: {parser.get_error(error)}")
                return False
        
        logger.info("Configuring TensorRT builder...")
        config = builder.create_builder_config()
        
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size_gb * (1 << 30))
        
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 mode enabled")
        
        profile = builder.create_optimization_profile()
        min_shape = (1, 3, resolution, resolution)
        opt_shape = (1, 3, resolution, resolution)
        max_shape = (1, 3, resolution, resolution)
        
        profile.set_shape("frame1", min_shape, opt_shape, max_shape)
        profile.set_shape("frame2", min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        logger.info("Building TensorRT engine... (this will take a while)")
        engine = builder.build_serialized_network(network, config)
        
        if engine is None:
            logger.error("Failed to build TensorRT engine")
            return False
        
        logger.info(f"Saving engine to {engine_path}")
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(engine)
        
        logger.info(f"Successfully built and saved TensorRT engine: {engine_path}")
        logger.info(f"Engine size: {engine_path.stat().st_size / (1024*1024):.2f} MB")
        return True
        
    except Exception as e:
        logger.error(f"Failed to build TensorRT engine: {e}")
        import traceback
        traceback.print_exc()
        return False


def compile_raft(
    resolution: int = 512,
    output_dir: str = "./models/temporal_net",
    device: str = "cuda",
    fp16: bool = True,
    workspace_size_gb: int = 4,
    force_rebuild: bool = False
):
    """
    Main function to compile RAFT model to TensorRT engine
    
    Args:
        resolution: Input resolution for the model (default: 512)
        output_dir: Directory to save the models (default: ./models/temporal_net)
        device: Device to use for export (default: cuda)
        fp16: Enable FP16 precision mode (default: True)
        workspace_size_gb: Maximum workspace size in GB (default: 4)
        force_rebuild: Force rebuild even if engine exists (default: False)
    """
    if not TENSORRT_AVAILABLE:
        logger.error("TensorRT is not available. Please install it first using:")
        logger.error("  python -m streamdiffusion.tools.install-tensorrt")
        return
    
    if not TORCHVISION_AVAILABLE:
        logger.error("torchvision is not available. Please install it first using:")
        logger.error("  pip install torchvision")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    onnx_path = output_path / "raft_small.onnx"
    engine_path = output_path / f"raft_small.trt"
    
    logger.info("="*80)
    logger.info("RAFT TensorRT Compilation")
    logger.info("="*80)
    logger.info(f"Output directory: {output_path.absolute()}")
    logger.info(f"ONNX path: {onnx_path}")
    logger.info(f"Engine path: {engine_path}")
    logger.info("="*80)
    
    if engine_path.exists() and not force_rebuild:
        logger.info(f"TensorRT engine already exists: {engine_path}")
        logger.info("Use --force_rebuild to rebuild it")
        return
    
    if not onnx_path.exists() or force_rebuild:
        logger.info("\n[Step 1/2] Exporting RAFT to ONNX...")
        if not export_raft_to_onnx(onnx_path, resolution, device):
            logger.error("Failed to export ONNX model")
            return
    else:
        logger.info(f"\n[Step 1/2] ONNX model already exists: {onnx_path}")
    
    logger.info("\n[Step 2/2] Building TensorRT engine...")
    if not build_tensorrt_engine(onnx_path, engine_path, resolution, fp16, workspace_size_gb):
        logger.error("Failed to build TensorRT engine")
        return
    
    logger.info("\n" + "="*80)
    logger.info("✓ Compilation completed successfully!")
    logger.info("="*80)
    logger.info(f"Engine path: {engine_path.absolute()}")
    logger.info("\nYou can now use this engine in TemporalNetTensorRTPreprocessor:")
    logger.info(f'  engine_path="{engine_path.absolute()}"')
    logger.info("="*80)


if __name__ == "__main__":
    fire.Fire(compile_raft)

