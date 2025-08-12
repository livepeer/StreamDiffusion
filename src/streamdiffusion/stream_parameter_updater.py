from typing import List, Optional, Dict, Tuple, Literal, Any, Callable
import torch
import torch.nn.functional as F
import gc

import logging
logger = logging.getLogger(__name__)

class CacheStats:
    """Helper class to track cache statistics"""
    def __init__(self):
        self.hits = 0
        self.misses = 0

    def record_hit(self):
        self.hits += 1

    def record_miss(self):
        self.misses += 1


class StreamParameterUpdater:
    def __init__(self, stream_diffusion, wrapper=None, normalize_prompt_weights: bool = True, normalize_seed_weights: bool = True):
        self.stream = stream_diffusion
        self.wrapper = wrapper  # Reference to wrapper for accessing pipeline structure
        self.normalize_prompt_weights = normalize_prompt_weights
        self.normalize_seed_weights = normalize_seed_weights
        # Prompt blending caches
        self._prompt_cache: Dict[int, Dict] = {}
        self._current_prompt_list: List[Tuple[str, float]] = []
        self._current_negative_prompt: str = ""
        self._prompt_cache_stats = CacheStats()

        # Seed blending caches
        self._seed_cache: Dict[int, Dict] = {}
        self._current_seed_list: List[Tuple[int, float]] = []
        self._seed_cache_stats = CacheStats()
        
        # Enhancement hooks (e.g., for IPAdapter)
        self._embedding_enhancers = []
        
        # IPAdapter embedding preprocessing
        self._embedding_preprocessors = []
        self._embedding_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._current_style_images: Dict[str, Any] = {}
        self._embedding_orchestrator = None
    def get_cache_info(self) -> Dict:
        """Get cache statistics for monitoring performance."""
        total_requests = self._prompt_cache_stats.hits + self._prompt_cache_stats.misses
        hit_rate = self._prompt_cache_stats.hits / total_requests if total_requests > 0 else 0

        total_seed_requests = self._seed_cache_stats.hits + self._seed_cache_stats.misses
        seed_hit_rate = self._seed_cache_stats.hits / total_seed_requests if total_seed_requests > 0 else 0

        return {
            "cached_prompts": len(self._prompt_cache),
            "cache_hits": self._prompt_cache_stats.hits,
            "cache_misses": self._prompt_cache_stats.misses,
            "hit_rate": f"{hit_rate:.2%}",
            "current_prompts": len(self._current_prompt_list),
            "cached_seeds": len(self._seed_cache),
            "seed_cache_hits": self._seed_cache_stats.hits,
            "seed_cache_misses": self._seed_cache_stats.misses,
            "seed_hit_rate": f"{seed_hit_rate:.2%}",
            "current_seeds": len(self._current_seed_list)
        }

    def clear_caches(self) -> None:
        """Clear all caches to free memory."""
        self._prompt_cache.clear()
        self._current_prompt_list.clear()
        self._current_negative_prompt = ""
        self._prompt_cache_stats = CacheStats()

        self._seed_cache.clear()
        self._current_seed_list.clear()
        self._seed_cache_stats = CacheStats()
        
        # Clear embedding caches
        self._embedding_cache.clear()
        self._current_style_images.clear()

    def get_normalize_prompt_weights(self) -> bool:
        """Get the current prompt weight normalization setting."""
        return self.normalize_prompt_weights

    def get_normalize_seed_weights(self) -> bool:
        """Get the current seed weight normalization setting."""
        return self.normalize_seed_weights
    
    def register_embedding_enhancer(self, enhancer_func, name: str = "unknown") -> None:
        """
        Register an embedding enhancer function that will be called after prompt blending.
        
        The enhancer function should have signature:
        enhancer_func(prompt_embeds: torch.Tensor, negative_prompt_embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        
        Args:
            enhancer_func: Function that takes (prompt_embeds, negative_prompt_embeds) and returns enhanced versions
            name: Optional name for the enhancer (for debugging)
        """
        self._embedding_enhancers.append((enhancer_func, name))
        # IMMEDIATELY apply enhancer to existing embeddings if they exist (fixes TensorRT timing issue)
        if hasattr(self.stream, 'prompt_embeds') and self.stream.prompt_embeds is not None:
            try:
                current_negative_embeds = getattr(self.stream, 'negative_prompt_embeds', None)
                enhanced_prompt_embeds, enhanced_negative_embeds = enhancer_func(
                    self.stream.prompt_embeds, current_negative_embeds
                )
                self.stream.prompt_embeds = enhanced_prompt_embeds
                if enhanced_negative_embeds is not None:
                    self.stream.negative_prompt_embeds = enhanced_negative_embeds
            except Exception as e:
                print(f"register_embedding_enhancer: Error applying '{name}' enhancer immediately: {e}")
                import traceback
                traceback.print_exc()
    
    def unregister_embedding_enhancer(self, enhancer_func) -> None:
        """Unregister a specific embedding enhancer function."""
        original_length = len(self._embedding_enhancers)
        self._embedding_enhancers = [(func, name) for func, name in self._embedding_enhancers if func != enhancer_func]
        removed_count = original_length - len(self._embedding_enhancers)

    
    def clear_embedding_enhancers(self) -> None:
        """Clear all embedding enhancers."""
        enhancer_count = len(self._embedding_enhancers)
        self._embedding_enhancers.clear()

    def register_embedding_preprocessor(self, preprocessor: Any, style_image_key: str) -> None:
        """
        Register an embedding preprocessor for parallel processing.
        
        Args:
            preprocessor: IPAdapterEmbeddingPreprocessor instance
            style_image_key: Unique key for the style image this preprocessor handles
        """
        if self._embedding_orchestrator is None:
            from .preprocessing.preprocessing_orchestrator import PreprocessingOrchestrator
            self._embedding_orchestrator = PreprocessingOrchestrator(
                device=self.stream.device,
                dtype=self.stream.dtype,
                max_workers=4
            )
        
        self._embedding_preprocessors.append((preprocessor, style_image_key))
    
    def unregister_embedding_preprocessor(self, style_image_key: str) -> None:
        """Unregister an embedding preprocessor by style image key."""
        original_count = len(self._embedding_preprocessors)
        self._embedding_preprocessors = [
            (preprocessor, key) for preprocessor, key in self._embedding_preprocessors 
            if key != style_image_key
        ]
        removed_count = original_count - len(self._embedding_preprocessors)
        
        # Clear cached embeddings for this key
        if style_image_key in self._embedding_cache:
            del self._embedding_cache[style_image_key]
        if style_image_key in self._current_style_images:
            del self._current_style_images[style_image_key]
    
    def update_style_image(self, style_image_key: str, style_image: Any, is_stream: bool = False) -> None:
        """
        Update a style image and trigger embedding preprocessing.
        
        Args:
            style_image_key: Unique key for the style image
            style_image: The style image (PIL Image, path, etc.)
            is_stream: If True, use pipelined processing (1-frame lag, high throughput)
                      If False, use synchronous processing (immediate results, lower throughput)
        """
        # Store the style image
        self._current_style_images[style_image_key] = style_image
        
        # Trigger preprocessing for this style image
        self._preprocess_style_image_parallel(style_image_key, style_image, is_stream)
    
    def _preprocess_style_image_parallel(self, style_image_key: str, style_image: Any, is_stream: bool = False) -> None:
        """
        Preprocessing for a specific style image with mode selection
        
        Args:
            style_image_key: Unique key for the style image
            style_image: The style image to process
            is_stream: If True, use pipelined processing; if False, use synchronous processing
        """
        if not self._embedding_preprocessors or self._embedding_orchestrator is None:
            return
        
        # Find preprocessors for this key
        relevant_preprocessors = [
            preprocessor for preprocessor, key in self._embedding_preprocessors 
            if key == style_image_key
        ]
        
        if not relevant_preprocessors:
            return
        
        # Choose processing mode based on is_stream parameter
        try:
            if is_stream:
                # Pipelined processing - optimized for throughput with 1-frame lag
                embedding_results = self._embedding_orchestrator.process_embedding_preprocessors_pipelined(
                    input_image=style_image,
                    embedding_preprocessors=relevant_preprocessors,
                    stream_width=self.stream.width,
                    stream_height=self.stream.height
                )
            else:
                # Synchronous processing - immediate results for discrete updates
                embedding_results = self._embedding_orchestrator.process_embedding_preprocessors(
                    input_image=style_image,
                    embedding_preprocessors=relevant_preprocessors,
                    stream_width=self.stream.width,
                    stream_height=self.stream.height
                )
            
            # Cache results for this style image key
            if embedding_results and embedding_results[0] is not None:
                self._embedding_cache[style_image_key] = embedding_results[0]
            else:
                # This is an error condition - we should always have results
                raise RuntimeError(f"_preprocess_style_image_parallel: Failed to generate embeddings for style image '{style_image_key}'")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
    
    def get_cached_embeddings(self, style_image_key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached embeddings for a style image key"""
        cached_result = self._embedding_cache.get(style_image_key, None)
        return cached_result


    def _normalize_weights(self, weights: List[float], normalize: bool) -> torch.Tensor:
        """Generic weight normalization helper"""
        weights_tensor = torch.tensor(weights, device=self.stream.device, dtype=self.stream.dtype)
        if normalize:
            weights_tensor = weights_tensor / weights_tensor.sum()
        return weights_tensor

    def _validate_index(self, index: int, item_list: List, operation_name: str) -> bool:
        """Generic index validation helper"""
        if not item_list:
            logger.warning(f"{operation_name}: Warning: No current item list")
            return False

        if index < 0 or index >= len(item_list):
            logger.warning(f"{operation_name}: Warning: Index {index} out of range (0-{len(item_list)-1})")
            return False

        return True

    def _reindex_cache(self, cache: Dict[int, Dict], removed_index: int) -> Dict[int, Dict]:
        """Generic cache reindexing helper after item removal"""
        new_cache = {}
        for cache_idx, cache_data in cache.items():
            if cache_idx < removed_index:
                new_cache[cache_idx] = cache_data
            elif cache_idx > removed_index:
                new_cache[cache_idx - 1] = cache_data
        return new_cache

    @torch.no_grad()
    def update_stream_params(
        self,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        delta: Optional[float] = None,
        t_index_list: Optional[List[int]] = None,
        seed: Optional[int] = None,
        prompt_list: Optional[List[Tuple[str, float]]] = None,
        negative_prompt: Optional[str] = None,
        prompt_interpolation_method: Literal["linear", "slerp"] = "slerp",
        normalize_prompt_weights: Optional[bool] = None,
        seed_list: Optional[List[Tuple[int, float]]] = None,
        seed_interpolation_method: Literal["linear", "slerp"] = "linear",
        normalize_seed_weights: Optional[bool] = None,
        controlnet_config: Optional[List[Dict[str, Any]]] = None,
        ipadapter_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update streaming parameters efficiently in a single call."""

        if num_inference_steps is not None:
            self.stream.scheduler.set_timesteps(num_inference_steps, self.stream.device)
            self.stream.timesteps = self.stream.scheduler.timesteps.to(self.stream.device)

        if num_inference_steps is not None and t_index_list is None:
            max_step = num_inference_steps - 1
            t_index_list = [min(t, max_step) for t in self.stream.t_list]

        if guidance_scale is not None:
            if self.stream.cfg_type == "none" and guidance_scale > 1.0:
                logger.warning("update_stream_params: Warning: guidance_scale > 1.0 with cfg_type='none' will have no effect")
            self.stream.guidance_scale = guidance_scale

        if delta is not None:
            self.stream.delta = delta

        if seed is not None:
            self._update_seed(seed)
        
        if normalize_prompt_weights is not None:
            self.normalize_prompt_weights = normalize_prompt_weights
            logger.info(f"update_stream_params: Prompt weight normalization set to {normalize_prompt_weights}")

        if normalize_seed_weights is not None:
            self.normalize_seed_weights = normalize_seed_weights
            logger.info(f"update_stream_params: Seed weight normalization set to {normalize_seed_weights}")

        # Handle prompt blending if prompt_list is provided
        if prompt_list is not None:
            self._update_blended_prompts(
                prompt_list=prompt_list,
                negative_prompt=negative_prompt or self._current_negative_prompt,
                prompt_interpolation_method=prompt_interpolation_method
            )

        # Handle seed blending if seed_list is provided
        if seed_list is not None:
            self._update_blended_seeds(
                seed_list=seed_list,
                interpolation_method=seed_interpolation_method
            )

        if t_index_list is not None:
            self._recalculate_timestep_dependent_params(t_index_list)

        # Handle ControlNet configuration updates
        if controlnet_config is not None:
            logger.info(f"update_stream_params: Updating ControlNet configuration with {len(controlnet_config)} controlnets")
            self._update_controlnet_config(controlnet_config)
        
        # Handle IPAdapter configuration updates
        if ipadapter_config is not None:
            logger.info(f"update_stream_params: Updating IPAdapter configuration")
            self._update_ipadapter_config(ipadapter_config)

    @torch.no_grad()
    def update_prompt_weights(
        self,
        prompt_weights: List[float],
        prompt_interpolation_method: Literal["linear", "slerp"] = "slerp"
    ) -> None:
        """Update weights for current prompt list without re-encoding prompts."""
        if not self._current_prompt_list:
            logger.warning("update_prompt_weights: Warning: No current prompt list to update weights for")
            return

        if len(prompt_weights) != len(self._current_prompt_list):
            logger.warning(f"update_prompt_weights: Warning: Weight count {len(prompt_weights)} doesn't match prompt count {len(self._current_prompt_list)}")
            return

        # Update the current prompt list with new weights
        updated_prompt_list = []
        for i, (prompt_text, _) in enumerate(self._current_prompt_list):
            updated_prompt_list.append((prompt_text, prompt_weights[i]))

        self._current_prompt_list = updated_prompt_list

        # Recompute blended embeddings with new weights
        self._apply_prompt_blending(prompt_interpolation_method)

    @torch.no_grad()
    def update_seed_weights(
        self,
        seed_weights: List[float],
        interpolation_method: Literal["linear", "slerp"] = "linear"
    ) -> None:
        """Update weights for current seed list without regenerating noise."""
        if not self._current_seed_list:
            logger.warning("update_seed_weights: Warning: No current seed list to update weights for")
            return

        if len(seed_weights) != len(self._current_seed_list):
            logger.warning(f"update_seed_weights: Warning: Weight count {len(seed_weights)} doesn't match seed count {len(self._current_seed_list)}")
            return

        # Update the current seed list with new weights
        updated_seed_list = []
        for i, (seed_value, _) in enumerate(self._current_seed_list):
            updated_seed_list.append((seed_value, seed_weights[i]))

        self._current_seed_list = updated_seed_list

        # Recompute blended noise with new weights
        self._apply_seed_blending(interpolation_method)

    @torch.no_grad()
    def _update_blended_prompts(
        self,
        prompt_list: List[Tuple[str, float]],
        negative_prompt: str = "",
        prompt_interpolation_method: Literal["linear", "slerp"] = "slerp"
    ) -> None:
        """Update prompt embeddings using multiple weighted prompts."""
        # Store current state
        self._current_prompt_list = prompt_list.copy()
        self._current_negative_prompt = negative_prompt

        # Encode any new prompts and cache them
        self._cache_prompt_embeddings(prompt_list, negative_prompt)

        # Apply blending
        self._apply_prompt_blending(prompt_interpolation_method)

    def _cache_prompt_embeddings(
        self,
        prompt_list: List[Tuple[str, float]],
        negative_prompt: str
    ) -> None:
        """Cache prompt embeddings for efficient reuse."""
        for idx, (prompt_text, weight) in enumerate(prompt_list):
            if idx not in self._prompt_cache or self._prompt_cache[idx]['text'] != prompt_text:
                # Cache miss - encode the prompt
                self._prompt_cache_stats.record_miss()
                encoder_output = self.stream.pipe.encode_prompt(
                    prompt=prompt_text,
                    device=self.stream.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=negative_prompt,
                )
                self._prompt_cache[idx] = {
                    'embed': encoder_output[0],
                    'text': prompt_text
                }
            else:
                # Cache hit
                self._prompt_cache_stats.record_hit()

    def _apply_prompt_blending(self, prompt_interpolation_method: Literal["linear", "slerp"]) -> None:
        """Apply weighted blending of cached prompt embeddings."""
        if not self._current_prompt_list:
            return

        embeddings = []
        weights = []

        for idx, (prompt_text, weight) in enumerate(self._current_prompt_list):
            if idx in self._prompt_cache:
                embeddings.append(self._prompt_cache[idx]['embed'])
                weights.append(weight)

        if not embeddings:
            logger.warning("_apply_prompt_blending: Warning: No cached embeddings found")
            return

        # Normalize weights
        weights = self._normalize_weights(weights, self.normalize_prompt_weights)

        # Apply interpolation
        if prompt_interpolation_method == "slerp" and len(embeddings) == 2:
            # Spherical linear interpolation for 2 prompts
            embed1, embed2 = embeddings[0], embeddings[1]
            t = weights[1].item()  # Use second weight as interpolation factor
            combined_embeds = self._slerp(embed1, embed2, t)
        else:
            # Linear interpolation (weighted average)
            combined_embeds = torch.zeros_like(embeddings[0])
            for embed, weight in zip(embeddings, weights):
                combined_embeds += weight * embed

        # Handle CFG properly - need to set both conditional and unconditional if using CFG
        if self.stream.cfg_type in ["full", "initialize"] and self.stream.guidance_scale > 1.0:
            # For CFG, prompt_embeds contains [uncond, cond] concatenated
            batch_size = self.stream.batch_size // 2 if self.stream.cfg_type == "full" else self.stream.batch_size

            # Get unconditional embeddings (empty prompt)
            uncond_output = self.stream.pipe.encode_prompt(
                prompt="",
                device=self.stream.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=self._current_negative_prompt,
            )
            uncond_embeds = uncond_output[0].repeat(batch_size, 1, 1)

            # Combine with conditional embeddings
            cond_embeds = combined_embeds.repeat(batch_size, 1, 1)
            final_prompt_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)
            final_negative_embeds = None  # CFG mode combines everything into prompt_embeds
        else:
            # No CFG, just use the blended embeddings
            final_prompt_embeds = combined_embeds.repeat(self.stream.batch_size, 1, 1)
            final_negative_embeds = None  # Will be set by enhancers if needed
        
        # Apply embedding enhancers (e.g., IPAdapter)
        if self._embedding_enhancers:
            for enhancer_func, enhancer_name in self._embedding_enhancers:
                try:
                    enhanced_prompt_embeds, enhanced_negative_embeds = enhancer_func(
                        final_prompt_embeds, final_negative_embeds
                    )
                    final_prompt_embeds = enhanced_prompt_embeds
                    if enhanced_negative_embeds is not None:
                        final_negative_embeds = enhanced_negative_embeds
                except Exception as e:
                    print(f"_apply_prompt_blending: Error in enhancer '{enhancer_name}': {e}")
                    import traceback
                    traceback.print_exc()
        
        # Set final embeddings on stream
        self.stream.prompt_embeds = final_prompt_embeds
        if final_negative_embeds is not None:
            self.stream.negative_prompt_embeds = final_negative_embeds

    def _slerp(self, embed1: torch.Tensor, embed2: torch.Tensor, t: float) -> torch.Tensor:
        """Spherical linear interpolation between two embeddings."""
        # Handle case where t is 0 or 1
        if t <= 0:
            return embed1
        if t >= 1:
            return embed2

        # SLERP on flattened embeddings but preserve original shape
        original_shape = embed1.shape
        flat1 = embed1.view(-1)
        flat2 = embed2.view(-1)

        # Normalize
        flat1_norm = F.normalize(flat1, dim=0)
        flat2_norm = F.normalize(flat2, dim=0)

        # Calculate angle
        dot_product = torch.clamp(torch.dot(flat1_norm, flat2_norm), -1.0, 1.0)
        theta = torch.acos(dot_product)

        # Handle parallel vectors
        if theta.abs() < 1e-6:
            result = (1 - t) * flat1 + t * flat2
        else:
            # SLERP formula
            sin_theta = torch.sin(theta)
            w1 = torch.sin((1 - t) * theta) / sin_theta
            w2 = torch.sin(t * theta) / sin_theta
            result = w1 * flat1 + w2 * flat2

        return result.view(original_shape)

    @torch.no_grad()
    def _update_blended_seeds(
        self,
        seed_list: List[Tuple[int, float]],
        interpolation_method: Literal["linear", "slerp"] = "linear"
    ) -> None:
        """Update seed tensors using multiple weighted seeds."""
        # Store current state
        self._current_seed_list = seed_list.copy()

        # Cache any new seed noise tensors
        self._cache_seed_noise(seed_list)

        # Apply blending
        self._apply_seed_blending(interpolation_method)

    def _cache_seed_noise(self, seed_list: List[Tuple[int, float]]) -> None:
        """Cache seed noise tensors for efficient reuse."""
        for idx, (seed_value, weight) in enumerate(seed_list):
            if idx not in self._seed_cache or self._seed_cache[idx]['seed'] != seed_value:
                # Cache miss - generate noise for the seed
                self._seed_cache_stats.record_miss()
                generator = torch.Generator(device=self.stream.device)
                generator.manual_seed(seed_value)

                noise = torch.randn(
                    (self.stream.batch_size, 4, self.stream.latent_height, self.stream.latent_width),
                    generator=generator,
                    device=self.stream.device,
                    dtype=self.stream.dtype
                )

                self._seed_cache[idx] = {
                    'noise': noise,
                    'seed': seed_value
                }
            else:
                # Cache hit
                self._seed_cache_stats.record_hit()

    def _apply_seed_blending(self, interpolation_method: Literal["linear", "slerp"]) -> None:
        """Apply weighted blending of cached seed noise tensors."""
        if not self._current_seed_list:
            return

        noise_tensors = []
        weights = []

        for idx, (seed_value, weight) in enumerate(self._current_seed_list):
            if idx in self._seed_cache:
                noise_tensors.append(self._seed_cache[idx]['noise'])
                weights.append(weight)

        if not noise_tensors:
            logger.warning("_apply_seed_blending: Warning: No cached noise tensors found")
            return

        # Normalize weights
        weights = self._normalize_weights(weights, self.normalize_seed_weights)

        # Apply interpolation
        if interpolation_method == "slerp" and len(noise_tensors) == 2:
            # Spherical linear interpolation for 2 seeds
            noise1, noise2 = noise_tensors[0], noise_tensors[1]
            t = weights[1].item()  # Use second weight as interpolation factor
            combined_noise = self._slerp_noise(noise1, noise2, t)
        else:
            # Linear interpolation (weighted average)
            combined_noise = torch.zeros_like(noise_tensors[0])
            for noise, weight in zip(noise_tensors, weights):
                combined_noise += weight * noise
            
            # Preserve noise magnitude when weights are normalized
            if self.normalize_seed_weights and len(noise_tensors) > 1:
                original_magnitude = torch.mean(torch.stack([torch.norm(noise) for noise in noise_tensors]))
                current_magnitude = torch.norm(combined_noise)
                if current_magnitude > 1e-8:  # Avoid division by zero
                    combined_noise = combined_noise * (original_magnitude / current_magnitude)

        # Update stream noise
        self.stream.init_noise = combined_noise
        self.stream.stock_noise = torch.zeros_like(self.stream.init_noise)

    def _slerp_noise(self, noise1: torch.Tensor, noise2: torch.Tensor, t: float) -> torch.Tensor:
        """Spherical linear interpolation between two noise tensors."""
        # Handle case where t is 0 or 1
        if t <= 0:
            return noise1
        if t >= 1:
            return noise2

        # SLERP on flattened noise but preserve original shape
        original_shape = noise1.shape
        flat1 = noise1.view(-1)
        flat2 = noise2.view(-1)

        # Normalize
        flat1_norm = F.normalize(flat1, dim=0)
        flat2_norm = F.normalize(flat2, dim=0)

        # Calculate angle
        dot_product = torch.clamp(torch.dot(flat1_norm, flat2_norm), -1.0, 1.0)
        theta = torch.acos(dot_product)

        # Handle parallel vectors
        if theta.abs() < 1e-6:
            result = (1 - t) * flat1 + t * flat2
        else:
            # SLERP formula
            sin_theta = torch.sin(theta)
            w1 = torch.sin((1 - t) * theta) / sin_theta
            w2 = torch.sin(t * theta) / sin_theta
            result = w1 * flat1 + w2 * flat2

        return result.view(original_shape)

    def _update_seed(self, seed: int) -> None:
        """Update the generator seed and regenerate seed-dependent tensors."""
        if self.stream.generator is None:
            print("update_stream_params: Warning: generator is None, cannot update seed")
            return

        # Store the current seed value
        self.stream.current_seed = seed

        # Update generator seed
        self.stream.generator.manual_seed(seed)

        # Regenerate init_noise tensor with new seed
        self.stream.init_noise = torch.randn(
            (self.stream.batch_size, 4, self.stream.latent_height, self.stream.latent_width),
            generator=self.stream.generator,
        ).to(device=self.stream.device, dtype=self.stream.dtype)

        # Reset stock_noise to match the new init_noise
        self.stream.stock_noise = torch.zeros_like(self.stream.init_noise)

    def _recalculate_timestep_dependent_params(self, t_index_list: List[int]) -> None:
        """Recalculate all parameters that depend on t_index_list."""
        self.stream.t_list = t_index_list

        self.stream.sub_timesteps = []
        for t in self.stream.t_list:
            self.stream.sub_timesteps.append(self.stream.timesteps[t])

        sub_timesteps_tensor = torch.tensor(
            self.stream.sub_timesteps, dtype=torch.long, device=self.stream.device
        )
        self.stream.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.stream.frame_bff_size if self.stream.use_denoising_batch else 1,
            dim=0,
        )

        c_skip_list = []
        c_out_list = []
        for timestep in self.stream.sub_timesteps:
            c_skip, c_out = self.stream._get_scheduler_scalings(timestep)
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.stream.c_skip = (
            torch.stack(c_skip_list)
            .view(len(self.stream.t_list), 1, 1, 1)
            .to(dtype=self.stream.dtype, device=self.stream.device)
        )
        self.stream.c_out = (
            torch.stack(c_out_list)
            .view(len(self.stream.t_list), 1, 1, 1)
            .to(dtype=self.stream.dtype, device=self.stream.device)
        )

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.stream.sub_timesteps:
            alpha_prod_t_sqrt = self.stream.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.stream.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)

        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.stream.t_list), 1, 1, 1)
            .to(dtype=self.stream.dtype, device=self.stream.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.stream.t_list), 1, 1, 1)
            .to(dtype=self.stream.dtype, device=self.stream.device)
        )
        self.stream.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.stream.frame_bff_size if self.stream.use_denoising_batch else 1,
            dim=0,
        )
        self.stream.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.stream.frame_bff_size if self.stream.use_denoising_batch else 1,
            dim=0,
        )

    def _regenerate_resolution_tensors(self) -> None:
        """This method is no longer used - resolution updates now restart the pipeline"""
        pass

    def _update_controlnet_inputs(self, width: int, height: int) -> None:
        """This method is no longer used - resolution updates now restart the pipeline"""
        pass

    def _recalculate_controlnet_inputs(self, width: int, height: int) -> None:
        """This method is no longer used - resolution updates now restart the pipeline"""
        pass

    @torch.no_grad()
    def update_prompt_at_index(
        self,
        index: int,
        new_prompt: str,
        prompt_interpolation_method: Literal["linear", "slerp"] = "slerp"
    ) -> None:
        """Update a single prompt at the specified index without re-encoding others."""
        if not self._validate_index(index, self._current_prompt_list, "update_prompt_at_index"):
            return

        # Update the prompt text while keeping the weight
        old_prompt, weight = self._current_prompt_list[index]
        self._current_prompt_list[index] = (new_prompt, weight)

        print(f"update_prompt_at_index: Updated prompt {index}: '{old_prompt[:30]}...' -> '{new_prompt[:30]}...'")

        # Cache the new prompt embedding
        self._cache_prompt_embeddings([(new_prompt, weight)], self._current_negative_prompt)

        # Update cache index to point to the new prompt
        if index in self._prompt_cache and self._prompt_cache[index]['text'] != new_prompt:
            # Find if this prompt is already cached elsewhere
            existing_cache_key = None
            for cache_idx, cache_data in self._prompt_cache.items():
                if cache_data['text'] == new_prompt:
                    existing_cache_key = cache_idx
                    break

            if existing_cache_key is not None:
                # Reuse existing cached embedding
                self._prompt_cache[index] = self._prompt_cache[existing_cache_key].copy()
                self._prompt_cache_stats.record_hit()
            else:
                # Encode new prompt
                self._prompt_cache_stats.record_miss()
                encoder_output = self.stream.pipe.encode_prompt(
                    prompt=new_prompt,
                    device=self.stream.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=self._current_negative_prompt,
                )
                self._prompt_cache[index] = {
                    'embed': encoder_output[0],
                    'text': new_prompt
                }

        # Recompute blended embeddings with updated prompt
        self._apply_prompt_blending(prompt_interpolation_method)

    @torch.no_grad()
    def get_current_prompts(self) -> List[Tuple[str, float]]:
        """Get the current prompt list with weights."""
        return self._current_prompt_list.copy()

    @torch.no_grad()
    def add_prompt(
        self,
        prompt: str,
        weight: float = 1.0,
        prompt_interpolation_method: Literal["linear", "slerp"] = "slerp"
    ) -> None:
        """Add a new prompt to the current list."""
        new_index = len(self._current_prompt_list)
        self._current_prompt_list.append((prompt, weight))

        print(f"add_prompt: Added prompt {new_index}: '{prompt[:30]}...' with weight {weight}")

        # Cache the new prompt
        encoder_output = self.stream.pipe.encode_prompt(
            prompt=prompt,
            device=self.stream.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=self._current_negative_prompt,
        )
        self._prompt_cache[new_index] = {
            'embed': encoder_output[0],
            'text': prompt
        }
        self._prompt_cache_stats.record_miss()

        # Recompute blended embeddings
        self._apply_prompt_blending(prompt_interpolation_method)

    @torch.no_grad()
    def remove_prompt_at_index(
        self,
        index: int,
        prompt_interpolation_method: Literal["linear", "slerp"] = "slerp"
    ) -> None:
        """Remove a prompt at the specified index."""
        if not self._validate_index(index, self._current_prompt_list, "remove_prompt_at_index"):
            return

        if len(self._current_prompt_list) <= 1:
            print("remove_prompt_at_index: Warning: Cannot remove last prompt")
            return

        # Remove from current list
        removed_prompt = self._current_prompt_list.pop(index)
        print(f"remove_prompt_at_index: Removed prompt {index}: '{removed_prompt[0][:30]}...'")

        # Remove from cache and reindex
        if index in self._prompt_cache:
            del self._prompt_cache[index]

        # Shift cache indices down
        self._prompt_cache = self._reindex_cache(self._prompt_cache, index)

        # Recompute blended embeddings
        self._apply_prompt_blending(prompt_interpolation_method)

    @torch.no_grad()
    def update_seed_at_index(
        self,
        index: int,
        new_seed: int,
        interpolation_method: Literal["linear", "slerp"] = "linear"
    ) -> None:
        """Update a single seed at the specified index without regenerating others."""
        if not self._validate_index(index, self._current_seed_list, "update_seed_at_index"):
            return

        # Update the seed value while keeping the weight
        old_seed, weight = self._current_seed_list[index]
        self._current_seed_list[index] = (new_seed, weight)

        print(f"update_seed_at_index: Updated seed {index}: {old_seed} -> {new_seed}")

        # Cache the new seed noise
        self._cache_seed_noise([(new_seed, weight)])

        # Update cache index to point to the new seed
        if index in self._seed_cache and self._seed_cache[index]['seed'] != new_seed:
            # Find if this seed is already cached elsewhere
            existing_cache_key = None
            for cache_idx, cache_data in self._seed_cache.items():
                if cache_data['seed'] == new_seed:
                    existing_cache_key = cache_idx
                    break

            if existing_cache_key is not None:
                # Reuse existing cached noise
                self._seed_cache[index] = self._seed_cache[existing_cache_key].copy()
                self._seed_cache_stats.record_hit()
            else:
                # Generate new noise
                self._seed_cache_stats.record_miss()
                generator = torch.Generator(device=self.stream.device)
                generator.manual_seed(new_seed)

                noise = torch.randn(
                    (self.stream.batch_size, 4, self.stream.latent_height, self.stream.latent_width),
                    generator=generator,
                    device=self.stream.device,
                    dtype=self.stream.dtype
                )

                self._seed_cache[index] = {
                    'noise': noise,
                    'seed': new_seed
                }

        # Recompute blended noise with updated seed
        self._apply_seed_blending(interpolation_method)

    @torch.no_grad()
    def get_current_seeds(self) -> List[Tuple[int, float]]:
        """Get the current seed list with weights."""
        return self._current_seed_list.copy()

    @torch.no_grad()
    def add_seed(
        self,
        seed: int,
        weight: float = 1.0,
        interpolation_method: Literal["linear", "slerp"] = "linear"
    ) -> None:
        """Add a new seed to the current list."""
        new_index = len(self._current_seed_list)
        self._current_seed_list.append((seed, weight))

        logger.info(f"add_seed: Added seed {new_index}: {seed} with weight {weight}")

        # Cache the new seed noise
        generator = torch.Generator(device=self.stream.device)
        generator.manual_seed(seed)

        noise = torch.randn(
            (self.stream.batch_size, 4, self.stream.latent_height, self.stream.latent_width),
            generator=generator,
            device=self.stream.device,
            dtype=self.stream.dtype
        )

        self._seed_cache[new_index] = {
            'noise': noise,
            'seed': seed
        }
        self._seed_cache_stats.record_miss()

        # Recompute blended noise
        self._apply_seed_blending(interpolation_method)

    @torch.no_grad()
    def remove_seed_at_index(
        self,
        index: int,
        interpolation_method: Literal["linear", "slerp"] = "linear"
    ) -> None:
        """Remove a seed at the specified index."""
        if not self._validate_index(index, self._current_seed_list, "remove_seed_at_index"):
            return

        if len(self._current_seed_list) <= 1:
            print("remove_seed_at_index: Warning: Cannot remove last seed")
            return

        # Remove from current list
        removed_seed = self._current_seed_list.pop(index)
        print(f"remove_seed_at_index: Removed seed {index}: {removed_seed[0]}")

        # Remove from cache and reindex
        if index in self._seed_cache:
            del self._seed_cache[index]

        # Shift cache indices down
        self._seed_cache = self._reindex_cache(self._seed_cache, index)

        # Recompute blended noise
        self._apply_seed_blending(interpolation_method)

    def _update_controlnet_config(self, desired_config: List[Dict[str, Any]]) -> None:
        """
        Update ControlNet configuration by diffing current vs desired state.
        
        Args:
            desired_config: Complete ControlNet configuration list defining the desired state.
                           Each dict contains: model_id, preprocessor, conditioning_scale, enabled, etc.
        """
        # Find the ControlNet pipeline (might be nested in IPAdapter)
        controlnet_pipeline = self._get_controlnet_pipeline()
        if not controlnet_pipeline:
            logger.warning(f"_update_controlnet_config: No ControlNet pipeline found")
            return
        
        current_config = self._get_current_controlnet_config()
        
        # Simple approach: detect what changed and apply minimal updates
        current_models = {i: getattr(cn, 'model_id', f'controlnet_{i}') for i, cn in enumerate(controlnet_pipeline.controlnets)}
        desired_models = {cfg['model_id']: cfg for cfg in desired_config}
        
        # Remove controlnets not in desired config
        for i in reversed(range(len(controlnet_pipeline.controlnets))):
            model_id = current_models.get(i, f'controlnet_{i}')
            if model_id not in desired_models:
                logger.info(f"_update_controlnet_config: Removing ControlNet {model_id}")
                controlnet_pipeline.remove_controlnet(i, immediate=False)
        
        # Add new controlnets and update existing ones
        for desired_cfg in desired_config:
            model_id = desired_cfg['model_id']
            existing_index = next((i for i, mid in current_models.items() if mid == model_id), None)
            
            if existing_index is None:
                # Add new controlnet
                logger.info(f"_update_controlnet_config: Adding ControlNet {model_id}")
                controlnet_pipeline.add_controlnet(desired_cfg, desired_cfg.get('control_image'), immediate=False)
            else:
                # Update existing controlnet
                if 'conditioning_scale' in desired_cfg:
                    current_scale = current_config[existing_index].get('conditioning_scale', 1.0)
                    desired_scale = desired_cfg['conditioning_scale']
                    
                    if current_scale != desired_scale:
                        logger.info(f"_update_controlnet_config: Updating {model_id} scale: {current_scale} → {desired_scale}")
                        controlnet_pipeline.update_controlnet_scale(existing_index, desired_scale)
                
                if 'preprocessor_params' in desired_cfg and hasattr(controlnet_pipeline, 'preprocessors') and controlnet_pipeline.preprocessors[existing_index]:
                    preprocessor = controlnet_pipeline.preprocessors[existing_index]
                    preprocessor.params.update(desired_cfg['preprocessor_params'])
                    for param_name, param_value in desired_cfg['preprocessor_params'].items():
                        if hasattr(preprocessor, param_name):
                            setattr(preprocessor, param_name, param_value)

    def _get_controlnet_pipeline(self):
        """
        Get the ControlNet pipeline from the pipeline structure (handles IPAdapter wrapping).
        
        Returns:
            ControlNet pipeline object or None if not found
        """
        # Check if stream is ControlNet pipeline directly
        if hasattr(self.stream, 'controlnets'):
            return self.stream
            
        # Check if stream has nested stream (IPAdapter wrapper)
        if hasattr(self.stream, 'stream') and hasattr(self.stream.stream, 'controlnets'):
            return self.stream.stream
        
        # Check if we have a wrapper reference and can access through it
        if self.wrapper and hasattr(self.wrapper, 'stream'):
            if hasattr(self.wrapper.stream, 'controlnets'):
                return self.wrapper.stream
            elif hasattr(self.wrapper.stream, 'stream') and hasattr(self.wrapper.stream.stream, 'controlnets'):
                return self.wrapper.stream.stream
        
        return None

    def _get_current_controlnet_config(self) -> List[Dict[str, Any]]:
        """
        Get current ControlNet configuration state.
        
        Returns:
            List of current ControlNet configurations
        """
        controlnet_pipeline = self._get_controlnet_pipeline()
        if not controlnet_pipeline or not hasattr(controlnet_pipeline, 'controlnets') or not controlnet_pipeline.controlnets:
            return []
        
        current_config = []
        for i, controlnet in enumerate(controlnet_pipeline.controlnets):
            model_id = getattr(controlnet, 'model_id', f'controlnet_{i}')
            scale = controlnet_pipeline.controlnet_scales[i] if hasattr(controlnet_pipeline, 'controlnet_scales') and i < len(controlnet_pipeline.controlnet_scales) else 1.0
            
            config = {
                'model_id': model_id,
                'conditioning_scale': scale,
                'preprocessor_params': getattr(controlnet_pipeline.preprocessors[i], 'params', {}) if hasattr(controlnet_pipeline, 'preprocessors') and controlnet_pipeline.preprocessors[i] else {}
            }
            current_config.append(config)
        
        return current_config

    def _update_ipadapter_config(self, desired_config: Dict[str, Any]) -> None:
        """
        Update IPAdapter configuration.
        
        Args:
            desired_config: IPAdapter configuration dict containing: 
                           ipadapter_model_path, image_encoder_path, style_image, scale, enabled, etc.
        """
        # Find the IPAdapter pipeline
        ipadapter_pipeline = self._get_ipadapter_pipeline()
        
        if not ipadapter_pipeline:
            logger.warning(f"_update_ipadapter_config: No IPAdapter pipeline found")
            return
        
        # Update scale if provided
        if 'scale' in desired_config:
            current_scale = getattr(ipadapter_pipeline, 'scale', 1.0)
            desired_scale = desired_config['scale']
            
            if current_scale != desired_scale:
                logger.info(f"_update_ipadapter_config: Updating scale: {current_scale} → {desired_scale}")
                ipadapter_pipeline.update_scale(desired_scale)
        
        # Update style image if provided
        if 'style_image' in desired_config:
            style_image = desired_config['style_image']
            if style_image is not None:
                logger.info(f"_update_ipadapter_config: Updating style image")
                ipadapter_pipeline.update_style_image(style_image)

    def _get_ipadapter_pipeline(self):
        """
        Get the IPAdapter pipeline from the pipeline structure (following ControlNet pattern).
        
        Returns:
            IPAdapter pipeline object or None if not found
        """
        # Check if stream is IPAdapter pipeline directly
        if hasattr(self.stream, 'ipadapter'):
            return self.stream
            
        # Check if stream has nested stream (ControlNet wrapper)
        if hasattr(self.stream, 'stream') and hasattr(self.stream.stream, 'ipadapter'):
            return self.stream.stream
        
        # Check if we have a wrapper reference and can access through it
        if self.wrapper and hasattr(self.wrapper, 'stream'):
            if hasattr(self.wrapper.stream, 'ipadapter'):
                return self.wrapper.stream
            elif hasattr(self.wrapper.stream, 'stream') and hasattr(self.wrapper.stream.stream, 'ipadapter'):
                return self.wrapper.stream.stream
        
        return None

    def _get_current_ipadapter_config(self) -> Optional[Dict[str, Any]]:
        """
        Get current IPAdapter configuration by introspecting the pipeline state.
        
        Returns:
            Current IPAdapter configuration dict or None if no IPAdapter
        """
        ipadapter_pipeline = self._get_ipadapter_pipeline()
        if not ipadapter_pipeline:
            return None
        
        config = {
            'scale': getattr(ipadapter_pipeline, 'scale', 1.0),
            'enabled': hasattr(ipadapter_pipeline, 'ipadapter') and ipadapter_pipeline.ipadapter is not None
        }
        
        # Add style image info if available
        if hasattr(ipadapter_pipeline, 'style_image') and ipadapter_pipeline.style_image:
            config['has_style_image'] = True
        else:
            config['has_style_image'] = False
            
        return config

