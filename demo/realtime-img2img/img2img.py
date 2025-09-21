import sys
import os
import logging

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

# Config system functions are now used only in main.py

import torch
from pydantic import BaseModel, Field
from PIL import Image
from typing import Optional

# Default values for pipeline parameters
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"

page_content = """<h1 class="text-3xl font-bold"><a href="https://github.com/livepeer/StreamDiffusion" target="_blank" class="text-blue-500 underline hover:no-underline">StreamDiffusion</a></h1>
<div class="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 mb-4">
    <div class="flex">
        <div class="flex-shrink-0">
            <svg class="h-5 w-5 text-yellow-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
            </svg>
        </div>
        <div class="ml-3">
            <div class="text-sm text-yellow-700">
                <p><strong>Development Tool Notice:</strong> This is an internal, vibe-coded development tool. It may change frequently and contain bugs. It is not supported.</p>
                <p>For production-level real-time research tools, use <a href="https://github.com/livepeer/stream-model-lab" target="_blank" class="text-blue-600 underline hover:no-underline">Livepeer Stream Model Lab</a></p>
            </div>
        </div>
    </div>
</div>

"""


class Pipeline:
    class Info(BaseModel):
        name: str = "StreamDiffusion"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        # negative_prompt: str = Field(
        #     default_negative_prompt,
        #     title="Negative Prompt",
        #     field="textarea",
        #     id="negative_prompt",
        # )
        resolution: str = Field(
            "512x512 (1:1)",
            title="Resolution",
            field="select",
            id="resolution",
            values=[
                # --- Square (1:1) ---
                "384x384 (1:1)",
                "512x512 (1:1)",
                "640x640 (1:1)",
                "704x704 (1:1)",
                "768x768 (1:1)",
                "896x896 (1:1)",
                "1024x1024 (1:1)",
                # --- Portrait ---
                "384x512 (3:4)",
                "512x768 (2:3)",
                "640x896 (5:7)",
                "768x1024 (3:4)",
                "576x1024 (9:16)",
                # --- Landscape ---
                "512x384 (4:3)",
                "768x512 (3:2)",
                "896x640 (7:5)",
                "1024x768 (4:3)",
                "1024x576 (16:9)"
            ]
        )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )

#TODO update naming convention to reflect the controlnet agnostic nature of the config system (pipeline_config instead of controlnet_config for example)
    def __init__(self, wrapper, config):
        """
        Initialize Pipeline with pre-created wrapper and config.
        
        Args:
            wrapper: Pre-created StreamDiffusionWrapper instance
            config: Configuration dictionary used to create the wrapper
        """
        
        # IPAdapter state tracking for optimization
        self._last_ipadapter_source_type = None
        self._last_ipadapter_source_data = None

        # Store the pre-created wrapper and config
        self.stream = wrapper
        self.config = config
        self.use_config = True
        
        # Extract pipeline configuration from config
        self.pipeline_mode = self.config.get('mode', 'img2img')
        self.has_controlnet = 'controlnets' in self.config and len(self.config['controlnets']) > 0
        self.has_ipadapter = 'ipadapters' in self.config and len(self.config['ipadapters']) > 0
        
        # Store config values for later use
        self.negative_prompt = self.config.get('negative_prompt', default_negative_prompt)
        self.guidance_scale = self.config.get('guidance_scale', 1.2)
        self.num_inference_steps = self.config.get('num_inference_steps', 50)

        # Update input_mode based on pipeline mode
        self.info = self.Info()
        if self.pipeline_mode == "txt2img":
            self.info.input_mode = "text"
        else:
            self.info.input_mode = "image"

        # Initialize pipeline parameters
        self.seed = 2
        self.guidance_scale = 1.1
        self.num_inference_steps = 50
        self.negative_prompt = default_negative_prompt
        
        # Store output type for frame conversion - always force "pt" for optimal performance
        self.output_type = "pt"

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        # Get input manager if available (passed from websocket handler)
        input_manager = getattr(params, 'input_manager', None)
        
        # Handle different modes
        if self.pipeline_mode == "txt2img":
            # Text-to-image mode
            
            # Handle ControlNet updates if enabled
            if self.has_controlnet:
                try:
                    stream_state = self.stream.get_stream_state()
                    current_cfg = stream_state.get('controlnet_config', [])
                except Exception:
                    current_cfg = []
                if current_cfg:
                    # Update control image for all configured ControlNets using input sources
                    for i in range(len(current_cfg)):
                        control_image = self._get_controlnet_input(input_manager, i, params.image)
                        if control_image is not None:
                            self.stream.update_control_image(index=i, image=control_image)
            
            # Handle IPAdapter updates if enabled
            if self.has_ipadapter:
                self._update_ipadapter_style_image(input_manager)
            
            # Generate output based on what's enabled
            if self.has_controlnet and not self.has_ipadapter:
                # ControlNet only: use base input for generation
                base_input = self._get_base_input(input_manager, params.image)
                output_image = self.stream(base_input)
            elif self.has_ipadapter and not self.has_controlnet:
                # IPAdapter only: no input image needed (style image handled separately)
                output_image = self.stream()
            elif self.has_controlnet and self.has_ipadapter:
                # Both enabled: use base input for generation (ControlNet + IPAdapter)
                base_input = self._get_base_input(input_manager, params.image)
                output_image = self.stream(base_input)
            else:
                # Pure txt2img: no image needed
                output_image = self.stream()
        else:
            # Image-to-image mode: use original logic
            
            # Handle ControlNet updates if enabled
            if self.has_controlnet:
                try:
                    stream_state = self.stream.get_stream_state()
                    current_cfg = stream_state.get('controlnet_config', [])
                except Exception:
                    current_cfg = []
                if current_cfg:
                    # Update control image for all configured ControlNets using input sources
                    for i in range(len(current_cfg)):
                        control_image = self._get_controlnet_input(input_manager, i, params.image)
                        if control_image is not None:
                            self.stream.update_control_image(index=i, image=control_image)
            
            # Handle IPAdapter updates if enabled
            if self.has_ipadapter:
                self._update_ipadapter_style_image(input_manager)
            
            # Generate output based on what's enabled
            if self.has_controlnet or self.has_ipadapter:
                # ControlNet and/or IPAdapter: use base input for img2img
                base_input = self._get_base_input(input_manager, params.image)
                output_image = self.stream(base_input)
            else:
                # Standard mode: handle tensor inputs (always from bytes_to_pt)
                base_input = self._get_base_input(input_manager, params.image)
                if isinstance(base_input, torch.Tensor):
                    # Direct tensor input - already preprocessed
                    output_image = self.stream(image=base_input)
                else:
                    # Fallback for PIL input - needs preprocessing
                    image_tensor = self.stream.preprocess_image(base_input)
                    output_image = self.stream(image=image_tensor)

        return output_image

    def _get_controlnet_input(self, input_manager, index: int, fallback_image):
        """
        Get input image for a specific ControlNet index.
        
        Args:
            input_manager: InputSourceManager instance (can be None)
            index: ControlNet index
            fallback_image: Fallback image if no specific source is configured
            
        Returns:
            Input image for the ControlNet or fallback
        """
        if input_manager:
            frame = input_manager.get_frame('controlnet', index)
            if frame is not None:
                return frame
        
        # Fallback to main image input
        return fallback_image
    
    def _get_ipadapter_input(self, input_manager):
        """
        Get input image for IPAdapter.
        
        Args:
            input_manager: InputSourceManager instance (can be None)
            
        Returns:
            Input image for IPAdapter or None
        """
        if input_manager:
            return input_manager.get_frame('ipadapter')
        return None
    
    def _update_ipadapter_style_image(self, input_manager):
        """
        Update IPAdapter style image from InputSourceManager.
        Only updates when source actually changes to avoid unnecessary processing.
        
        Args:
            input_manager: InputSourceManager instance (can be None)
        """
        if not input_manager or not self.has_ipadapter:
            return
            
        try:
            # Get current source info to check if it changed
            source_info = input_manager.get_source_info('ipadapter')
            current_source_type = source_info.get('source_type')
            current_source_data = source_info.get('source_data')
            is_stream = source_info.get('is_stream', False)
            
            # Check if source changed (for static images, only update when source changes)
            source_changed = (
                current_source_type != self._last_ipadapter_source_type or
                current_source_data != self._last_ipadapter_source_data
            )
            
            # For streaming sources (webcam/video), always get fresh frame
            # For static sources (uploaded image), only update when source changes
            should_update = is_stream or source_changed
            
            if not should_update:
                return  # No update needed - static source unchanged
            
            # Get IPAdapter style image from input source manager
            ipadapter_frame = input_manager.get_frame('ipadapter')
            
            if ipadapter_frame is not None:
                import torch
                
                # Use tensor directly - update_style_image expects torch tensor
                if isinstance(ipadapter_frame, torch.Tensor):
                    try:
                        # Update IPAdapter with tensor and stream configuration
                        self.stream.update_style_image(ipadapter_frame, is_stream=is_stream)
                        self.stream.update_stream_params(ipadapter_config={'is_stream': is_stream})
                        
                        # Force prompt re-encoding to apply new style image embeddings
                        # This is critical because IPAdapter embedding hook only runs during prompt encoding
                        try:
                            state = self.stream.get_stream_state()
                            current_prompts = state.get('prompt_list', [])
                            if current_prompts:
                                self.stream.update_prompt(current_prompts, prompt_interpolation_method="slerp")
                        except Exception as e:
                            logging.exception(f"_update_ipadapter_style_image: Failed to force prompt re-encoding: {e}")
                        
                        
                        # Update tracking variables only on successful update
                        self._last_ipadapter_source_type = current_source_type
                        self._last_ipadapter_source_data = current_source_data
                        
                    except Exception as e:
                        logging.exception(f"_update_ipadapter_style_image: Failed to update IPAdapter: {e}")
                else:
                    logging.warning("_update_ipadapter_style_image: IPAdapter frame is not a tensor, skipping style image update")
        except Exception as e:
            logging.exception(f"_update_ipadapter_style_image: Error updating IPAdapter style image: {e}")
    
    def _get_base_input(self, input_manager, fallback_image):
        """
        Get input image for base pipeline.
        
        Args:
            input_manager: InputSourceManager instance (can be None)
            fallback_image: Fallback image if no specific source is configured
            
        Returns:
            Input image for base pipeline or fallback
        """
        if input_manager:
            frame = input_manager.get_frame('base')
            if frame is not None:
                return frame
        
        # Fallback to main image input
        return fallback_image

    def update_ipadapter_config(self, scale: float = None, style_image: Image.Image = None) -> bool:
        """
        Update IPAdapter configuration in real-time using direct methods
        
        Args:
            scale: New IPAdapter scale value (optional)
            style_image: New style image (PIL Image, optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.has_ipadapter:
            return False
            
        if scale is None and style_image is None:
            return False  # Nothing to update
            
        try:
            # Update scale via unified config system (no direct method needed)
            if scale is not None:
                self.stream.update_stream_params(ipadapter_config={'scale': scale})
            
            # Update style image via direct method
            if style_image is not None:
                self.stream.update_style_image(style_image)
                
            return True
        except Exception as e:
            return False

    def update_ipadapter_scale(self, scale: float) -> bool:
        """Update IPAdapter scale - convenience method"""
        return self.update_ipadapter_config(scale=scale)

    def update_ipadapter_style_image(self, style_image: Image.Image) -> bool:
        """Update IPAdapter style image - convenience method"""
        return self.update_ipadapter_config(style_image=style_image)

    def update_ipadapter_weight_type(self, weight_type: str) -> bool:
        """Update IPAdapter weight type in real-time"""
        if not self.has_ipadapter:
            return False
            
        try:
            # Use unified updater on wrapper
            if hasattr(self.stream, 'update_stream_params'):
                self.stream.update_stream_params(ipadapter_config={ 'weight_type': weight_type })
                return True
            # Should not reach here in normal operation
            return False
        except Exception as e:
            return False

    def get_ipadapter_info(self) -> dict:
        """
        Get current IPAdapter information
        
        Returns:
            dict: IPAdapter information including scale, model info, etc.
        """
        info = {
            "enabled": self.has_ipadapter,
            "scale": 1.0,
            "weight_type": "linear",
            "model_path": None,
            "style_image_set": False
        }
        
        if self.has_ipadapter and self.config and 'ipadapters' in self.config:
            # Get info from first IPAdapter config
            if len(self.config['ipadapters']) > 0:
                ipadapter_config = self.config['ipadapters'][0]
                info["scale"] = ipadapter_config.get('scale', 1.0)
                info["weight_type"] = ipadapter_config.get('weight_type', 'linear')
                info["model_path"] = ipadapter_config.get('ipadapter_model_path')
                info["style_image_set"] = 'style_image' in ipadapter_config
                
        # Get current runtime state from wrapper's public API
        try:
            if hasattr(self.stream, 'get_stream_state'):
                stream_state = self.stream.get_stream_state()
                ipadapter_runtime_config = stream_state.get('ipadapter_config', {})
                if ipadapter_runtime_config:
                    info["scale"] = ipadapter_runtime_config.get('scale', info.get("scale", 1.0))
                    info["weight_type"] = ipadapter_runtime_config.get('weight_type', info.get("weight_type", 'linear'))
        except Exception:
            pass  # Use defaults from config if wrapper method fails
            
        return info

    def update_stream_params(self, **kwargs):
        """
        Update streaming parameters using the consolidated API
        
        Args:
            **kwargs: All parameters supported by StreamDiffusionWrapper.update_stream_params()
                     including controlnet_config, guidance_scale, delta, etc.
        """
        return self.stream.update_stream_params(**kwargs)
