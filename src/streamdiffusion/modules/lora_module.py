from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union, Literal
import torch
from pathlib import Path
import threading
import logging

from ..preprocessing.orchestrator_user import OrchestratorUser

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for a single LoRA."""
    lora_path: str
    adapter_name: Optional[str] = None
    scale: float = 1.0
    enabled: bool = True
    lora_type: Optional[Literal["text_encoder", "unet", "both"]] = None
    # Additional metadata
    display_name: Optional[str] = None
    description: Optional[str] = None


class LoRAModule(OrchestratorUser):
    """LoRA module providing comprehensive LoRA management and hotswapping."""
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        
        # State management
        self.loras: List[LoRAConfig] = []
        self.loaded_adapters: Dict[str, str] = {}  # adapter_name -> lora_path
        self._collections_lock = threading.RLock()
        
        # Pipeline reference (set during install)
        self._stream = None
        self._pipe = None
        
        # LoRA type detection
        self._lora_type_cache: Dict[str, str] = {}
        
        # Offline fallback support
        self._candidate_weight_names = (
            "pytorch_lora_weights.safetensors",
            "pytorch_lora_weights.bin", 
            "diffusion_pytorch_model.safetensors",
            "adapter_model.safetensors",
            "lora.safetensors",
        )

    def install(self, stream) -> None:
        """Install LoRA module into the pipeline."""
        self._stream = stream
        self._pipe = stream.pipe
        
        # Attach orchestrator for consistency
        self.attach_orchestrator(stream)
        
        # No hooks needed - LoRAs work transparently after loading
        # State management is handled at the module level
        logger.info("install: LoRA module installed successfully")

    def _detect_lora_type(self, lora_path: str) -> str:
        """Detect LoRA type from file content - text_encoder, unet, or both."""
        if lora_path in self._lora_type_cache:
            return self._lora_type_cache[lora_path]
            
        try:
            # Handle both file paths and HuggingFace model IDs
            if Path(lora_path).exists():
                # Local file - try to load and inspect
                try:
                    import safetensors.torch
                    if lora_path.endswith('.safetensors'):
                        lora_weights = safetensors.torch.load_file(lora_path, device='cpu')
                    else:
                        lora_weights = torch.load(lora_path, map_location='cpu')
                except Exception as e:
                    # If we can't load the weights, assume both
                    logger.warning(f"_detect_lora_type: Could not load weights from {lora_path}: {e}. Assuming 'both' type.")
                    lora_type = 'both'
                    self._lora_type_cache[lora_path] = lora_type
                    return lora_type
            else:
                # HuggingFace model ID - assume both for unknown models
                lora_type = 'both'
                logger.info(f"_detect_lora_type: Assuming 'both' type for HuggingFace model: {lora_path}")
                self._lora_type_cache[lora_path] = lora_type
                return lora_type
            
            # Check for text encoder vs unet patterns
            text_encoder_keys = [k for k in lora_weights.keys() if 'text_model' in k or 'text_encoder' in k or 'lora_te' in k]
            unet_keys = [k for k in lora_weights.keys() if 'unet' in k or 'diffusion_model' in k or 'lora_unet' in k]
            
            if text_encoder_keys and not unet_keys:
                lora_type = 'text_encoder'
                logger.info(f"_detect_lora_type: Detected text encoder LoRA from weight patterns in {lora_path}")
            elif unet_keys and not text_encoder_keys:
                lora_type = 'unet'
                logger.info(f"_detect_lora_type: Detected UNet LoRA from weight patterns in {lora_path}")
            elif unet_keys and text_encoder_keys:
                lora_type = 'both'
                logger.info(f"_detect_lora_type: Detected both text encoder and UNet LoRA from weight patterns in {lora_path}")
            else:
                lora_type = 'unknown'
                logger.info(f"_detect_lora_type: Detected unknown LoRA from weight patterns in {lora_path}")
        except Exception as e:
            logger.warning(f"_detect_lora_type: Failed to detect LoRA type for {lora_path}: {e}. Assuming 'both' type.")
            lora_type = 'both'
            
        self._lora_type_cache[lora_path] = lora_type
        return lora_type

    def _load_lora_with_offline_fallback(self, lora_path: str, adapter_name: Optional[str] = None, **kwargs) -> bool:
        """Load LoRA weights with offline fallback support."""
        try:
            logger.debug(f"_load_lora_with_offline_fallback: Trying to load {lora_path} with adapter_name={adapter_name}")
            self._pipe.load_lora_weights(lora_path, adapter_name=adapter_name, **kwargs)
            logger.info(f"_load_lora_with_offline_fallback: Successfully loaded {lora_path}")
            return True
        except Exception as e:
            message = str(e)
            logger.debug(f"_load_lora_with_offline_fallback: Initial load failed: {e}")
            is_offline_weight_error = isinstance(e, ValueError) and "must specify a `weight_name`" in message
            if not is_offline_weight_error:
                logger.error(f"_load_lora_with_offline_fallback: Failed to load LoRA {lora_path}: {e}")
                return False

        # Try offline fallback with common weight names
        logger.debug(f"_load_lora_with_offline_fallback: Trying offline fallback for {lora_path}")
        last_err: Optional[Exception] = None
        for weight_name in self._candidate_weight_names:
            try:
                logger.debug(f"_load_lora_with_offline_fallback: Trying weight_name={weight_name}")
                self._pipe.load_lora_weights(
                    lora_path,
                    adapter_name=adapter_name,
                    weight_name=weight_name,
                    **kwargs
                )
                logger.info(f"_load_lora_with_offline_fallback: Successfully loaded LoRA {lora_path} with weight_name={weight_name}")
                return True
            except Exception as e:
                logger.debug(f"_load_lora_with_offline_fallback: Failed with weight_name={weight_name}: {e}")
                last_err = e
                continue

        if last_err is not None:
            logger.error(f"_load_lora_with_offline_fallback: All fallback attempts failed for {lora_path}: {last_err}")
        return False

    def add_lora(self, config: LoRAConfig) -> bool:
        """Add a new LoRA to the pipeline."""
        with self._collections_lock:
            try:
                # 1. Validate LoRA file exists or is valid HF model ID
                if not (Path(config.lora_path).exists() or '/' in config.lora_path):
                    logger.error(f"add_lora: LoRA path does not exist and is not a valid HF model ID: {config.lora_path}")
                    return False
                
                # 2. Detect LoRA type if not specified
                if config.lora_type is None:
                    config.lora_type = self._detect_lora_type(config.lora_path)
                    type_description = self._get_type_description(config.lora_type)
                    logger.info(f"add_lora: Detected LoRA type: {config.lora_type} ({type_description}) for {config.lora_path}")
                else:
                    logger.info(f"add_lora: Using specified LoRA type: {config.lora_type} for {config.lora_path}")
                
                # 2.5. Check for TensorRT compatibility
                is_tensorrt = self._is_tensorrt_acceleration()
                logger.info(f"add_lora: TensorRT detection: {is_tensorrt}, LoRA type: {config.lora_type}")
                if is_tensorrt and config.lora_type == 'unet':
                    print("=" * 80)
                    print("TENSORRT COMPATIBILITY WARNING")
                    print("=" * 80)
                    print(f"Pure UNet LoRAs are NOT supported with TensorRT acceleration!")
                    print(f"LoRA: {config.lora_path}")
                    print(f"Detected Type: {config.lora_type}")
                    print(f"Only text_encoder and 'both' type LoRAs are supported with TensorRT pipelines.")
                    print("=" * 80)
                    print("This LoRA will NOT be loaded to prevent pipeline errors.")
                    print("=" * 80)
                    return False
                
                # 3. Generate adapter name if not provided
                if config.adapter_name is None:
                    # Get existing adapter names from pipeline if possible
                    existing_adapters = set()
                    if hasattr(self._pipe, 'get_list_adapters'):
                        try:
                            existing_adapters.update(self._pipe.get_list_adapters())
                            logger.debug(f"add_lora: Found existing adapters in pipeline: {existing_adapters}")
                        except Exception as e:
                            logger.debug(f"add_lora: Could not get existing adapters: {e}")
                    
                    # Also check our internal tracking
                    existing_adapters.update(self.loaded_adapters.keys())
                    
                    # Generate unique adapter name
                    import time
                    timestamp = int(time.time() * 1000) % 10000  # Last 4 digits of timestamp
                    base_name = f"lora_{timestamp}"
                    config.adapter_name = base_name
                    
                    # Ensure uniqueness
                    counter = 0
                    while config.adapter_name in existing_adapters:
                        counter += 1
                        config.adapter_name = f"{base_name}_{counter}"
                    
                    logger.info(f"add_lora: Generated unique adapter name: {config.adapter_name}")
                
                # 4. Load LoRA weights using pipe.load_lora_weights
                try:
                    logger.debug(f"add_lora: Loading LoRA weights: {config.lora_path} with adapter_name: {config.adapter_name}")
                    self._pipe.load_lora_weights(config.lora_path, adapter_name=config.adapter_name)
                    logger.info(f"add_lora: Successfully loaded LoRA weights")
                except Exception as e:
                    logger.debug(f"add_lora: Failed to load LoRA weights: {e}")
                    # Try offline fallback
                    if not self._load_lora_with_offline_fallback(config.lora_path, config.adapter_name):
                        return False
                
                # 5. Set adapter scale if supported and enabled
                if config.enabled and hasattr(self._pipe, 'set_adapters'):
                    try:
                        # Get current adapters
                        current_adapters = []
                        current_scales = []
                        
                        for lora in self.loras:
                            if lora.enabled:
                                current_adapters.append(lora.adapter_name)
                                current_scales.append(lora.scale)
                        
                        # Add new adapter
                        current_adapters.append(config.adapter_name)
                        current_scales.append(config.scale)
                        
                        logger.debug(f"add_lora: Calling set_adapters with adapters={current_adapters}, adapter_weights={current_scales}")
                        self._pipe.set_adapters(current_adapters, adapter_weights=current_scales)
                        logger.debug(f"add_lora: Successfully set adapter weights")
                        logger.info(f"add_lora: Set adapter scales: {dict(zip(current_adapters, current_scales))}")
                    except Exception as e:
                        logger.warning(f"add_lora: Failed to set adapter scale: {e}")
                
                # 6. Add to internal state
                self.loras.append(config)
                self.loaded_adapters[config.adapter_name] = config.lora_path
                
                logger.info(f"add_lora: Successfully added LoRA {config.lora_path} as {config.adapter_name}")
                return True
                
            except Exception as e:
                logger.error(f"add_lora: Failed to add LoRA {config.lora_path}: {e}")
                return False

    def remove_lora(self, index: int) -> bool:
        """Remove a LoRA from the pipeline."""
        with self._collections_lock:
            try:
                # 1. Validate index
                if index < 0 or index >= len(self.loras):
                    logger.error(f"remove_lora: Invalid index {index}, valid range: 0-{len(self.loras)-1}")
                    return False
                
                lora_config = self.loras[index]
                
                # 2. Unload LoRA via pipe.unload_lora_weights()
                if hasattr(self._pipe, 'unload_lora_weights'):
                    try:
                        self._pipe.unload_lora_weights(lora_config.adapter_name)
                        logger.info(f"remove_lora: Unloaded LoRA adapter {lora_config.adapter_name}")
                    except Exception as e:
                        logger.warning(f"remove_lora: Failed to unload LoRA adapter {lora_config.adapter_name}: {e}")
                
                # 3. Remove from internal state
                removed_lora = self.loras.pop(index)
                if removed_lora.adapter_name in self.loaded_adapters:
                    del self.loaded_adapters[removed_lora.adapter_name]
                
                # 4. Update remaining adapter scales
                if hasattr(self._pipe, 'set_adapters'):
                    try:
                        current_adapters = []
                        current_scales = []
                        
                        for lora in self.loras:
                            if lora.enabled:
                                current_adapters.append(lora.adapter_name)
                                current_scales.append(lora.scale)
                        
                        if current_adapters:
                            logger.debug(f"remove_lora: Calling set_adapters with adapters={current_adapters}, adapter_weights={current_scales}")
                            self._pipe.set_adapters(current_adapters, adapter_weights=current_scales)
                        else:
                            # Disable all adapters if none remain
                            logger.debug(f"remove_lora: Disabling all adapters")
                            self._pipe.set_adapters([], adapter_weights=[])
                        
                        logger.info(f"remove_lora: Updated adapter scales after removal")
                    except Exception as e:
                        logger.warning(f"remove_lora: Failed to update adapter scales: {e}")
                
                logger.info(f"remove_lora: Successfully removed LoRA at index {index}")
                return True
                
            except Exception as e:
                logger.error(f"remove_lora: Failed to remove LoRA at index {index}: {e}")
                return False

    def update_lora_scale(self, index: int, scale: float) -> bool:
        """Update LoRA scale at runtime."""
        logger.debug(f"update_lora_scale: Called with index={index}, scale={scale}")
        with self._collections_lock:
            try:
                # 1. Validate index and scale
                if index < 0 or index >= len(self.loras):
                    logger.error(f"update_lora_scale: Invalid index {index}, valid range: 0-{len(self.loras)-1}")
                    return False
                
                if scale < 0.0:
                    logger.error(f"update_lora_scale: Invalid scale {scale}, must be >= 0.0")
                    return False
                
                logger.debug(f"update_lora_scale: Before update - LoRA {index} scale was {self.loras[index].scale}")
                
                # 2. Update internal scale
                old_scale = self.loras[index].scale
                self.loras[index].scale = scale
                
                logger.debug(f"update_lora_scale: After internal update - LoRA {index} scale is now {self.loras[index].scale}")
                
                # 3. Apply new scale via pipe.set_adapters()
                logger.debug(f"update_lora_scale: Pipeline type: {type(self._pipe)}")
                logger.debug(f"update_lora_scale: Pipeline has set_adapters: {hasattr(self._pipe, 'set_adapters')}")
                if hasattr(self._pipe, 'set_adapters'):
                    try:
                        current_adapters = []
                        current_scales = []
                        
                        for i, lora in enumerate(self.loras):
                            if lora.enabled:
                                current_adapters.append(lora.adapter_name)
                                current_scales.append(lora.scale)
                                logger.debug(f"update_lora_scale: Including LoRA {i} ({lora.adapter_name}) with scale {lora.scale}")
                        
                        logger.debug(f"update_lora_scale: Calling set_adapters with adapters={current_adapters}, adapter_weights={current_scales}")
                        self._pipe.set_adapters(current_adapters, adapter_weights=current_scales)
                        logger.debug(f"update_lora_scale: set_adapters call completed successfully")
                        logger.info(f"update_lora_scale: Updated scale for index {index} to {scale}")
                    except Exception as e:
                        logger.error(f"update_lora_scale: Exception in set_adapters: {e}")
                        logger.warning(f"update_lora_scale: Failed to apply scale update: {e}")
                        return False
                else:
                    logger.warning(f"update_lora_scale: Pipeline does not have set_adapters method")
                
                return True
                
            except Exception as e:
                logger.error(f"update_lora_scale: Exception in main try block: {e}")
                return False

    def update_lora_enabled(self, index: int, enabled: bool) -> bool:
        """Enable/disable LoRA at runtime."""
        with self._collections_lock:
            try:
                # 1. Validate index
                if index < 0 or index >= len(self.loras):
                    logger.error(f"update_lora_enabled: Invalid index {index}, valid range: 0-{len(self.loras)-1}")
                    return False
                
                # 2. Update enabled state
                self.loras[index].enabled = enabled
                
                # 3. Apply changes via pipe.set_adapters()
                if hasattr(self._pipe, 'set_adapters'):
                    try:
                        current_adapters = []
                        current_scales = []
                        
                        for lora in self.loras:
                            if lora.enabled:
                                current_adapters.append(lora.adapter_name)
                                current_scales.append(lora.scale)
                        
                        logger.debug(f"update_lora_enabled: Calling set_adapters with adapters={current_adapters}, adapter_weights={current_scales}")
                        self._pipe.set_adapters(current_adapters, adapter_weights=current_scales)
                        logger.info(f"update_lora_enabled: Updated enabled state for index {index} to {enabled}")
                    except Exception as e:
                        logger.warning(f"update_lora_enabled: Failed to apply enabled state update: {e}")
                        return False
                
                return True
                
            except Exception as e:
                logger.error(f"update_lora_enabled: Failed to update enabled state for index {index}: {e}")
                return False

    def update_config(self, config: List[Dict[str, Any]]) -> None:
        """Update LoRA configuration from wrapper."""
        with self._collections_lock:
            try:
                # Convert dict configs to LoRAConfig objects
                desired_configs = []
                for cfg_dict in config:
                    lora_config = LoRAConfig(**cfg_dict)
                    desired_configs.append(lora_config)
                
                # Simple approach: clear all and reload
                # More sophisticated diffing could be implemented later
                
                # Remove all current LoRAs
                while self.loras:
                    self.remove_lora(0)
                
                # Add all desired LoRAs
                for lora_config in desired_configs:
                    self.add_lora(lora_config)
                
                logger.info(f"update_config: Updated configuration with {len(desired_configs)} LoRAs")
                
            except Exception as e:
                logger.error(f"update_config: Failed to update configuration: {e}")

    def get_loaded_loras_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about loaded LoRAs."""
        with self._collections_lock:
            return [
                {
                    'index': i,
                    'lora_path': lora.lora_path,
                    'adapter_name': lora.adapter_name,
                    'scale': lora.scale,
                    'enabled': lora.enabled,
                    'lora_type': lora.lora_type,
                    'display_name': lora.display_name,
                }
                for i, lora in enumerate(self.loras)
            ]

    def get_lora_state(self) -> Dict[str, Any]:
        """Get complete LoRA module state for debugging."""
        with self._collections_lock:
            return {
                'loaded_loras': len(self.loras),
                'enabled_loras': sum(1 for lora in self.loras if lora.enabled),
                'total_scales': [lora.scale for lora in self.loras],
                'lora_types': [lora.lora_type for lora in self.loras],
                'loaded_adapters': dict(self.loaded_adapters),
            }

    def get_lora_type_info(self, lora_path: str) -> Dict[str, Any]:
        """
        Get detailed LoRA type information for a specific LoRA.
        
        Args:
            lora_path: Path to the LoRA file or HuggingFace model ID
            
        Returns:
            Dictionary containing type information and detection details
        """
        lora_type = self._detect_lora_type(lora_path)
        
        # Check if this LoRA is currently loaded
        loaded_info = None
        with self._collections_lock:
            for lora in self.loras:
                if lora.lora_path == lora_path:
                    loaded_info = {
                        'is_loaded': True,
                        'adapter_name': lora.adapter_name,
                        'scale': lora.scale,
                        'enabled': lora.enabled,
                        'display_name': lora.display_name,
                        'description': lora.description
                    }
                    break
        
        if loaded_info is None:
            loaded_info = {'is_loaded': False}
        
        return {
            'lora_path': lora_path,
            'detected_type': lora_type,
            'type_description': self._get_type_description(lora_type),
            'is_cached': lora_path in self._lora_type_cache,
            'loaded_info': loaded_info
        }
    
    def _get_type_description(self, lora_type: str) -> str:
        """Get human-readable description of LoRA type."""
        descriptions = {
            'text_encoder': 'Text Encoder LoRA - affects only text processing',
            'unet': 'UNet LoRA - affects only the diffusion model',
            'both': 'Both Text Encoder and UNet LoRA - affects text processing and diffusion model',
            'unknown': 'Unknown LoRA type'
        }
        return descriptions.get(lora_type, f'Unknown LoRA type: {lora_type}')
    
    def _is_tensorrt_acceleration(self) -> bool:
        """Check if the pipeline is using TensorRT acceleration."""
        if not self._stream:
            logger.info("_is_tensorrt_acceleration: No stream available")
            return False
        
        # Check wrapper's acceleration setting
        if hasattr(self._stream, '_param_updater') and self._stream._param_updater.wrapper:
            wrapper = self._stream._param_updater.wrapper
            acceleration = getattr(wrapper, '_acceleration', None)
            logger.info(f"_is_tensorrt_acceleration: Wrapper acceleration: {acceleration}")
            return acceleration == 'tensorrt'
        
        logger.info("_is_tensorrt_acceleration: No wrapper available")
        return False
    