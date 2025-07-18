from typing import List, Optional, Dict, Tuple, Literal, Any, Callable
import torch
import torch.nn.functional as F


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
    def __init__(self, stream_diffusion, normalize_prompt_weights: bool = True, normalize_seed_weights: bool = True):
        self.stream = stream_diffusion
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

    def set_normalize_prompt_weights(self, normalize: bool) -> None:
        """Set whether to normalize prompt weights in blending operations."""
        self.normalize_prompt_weights = normalize
        print(f"set_normalize_prompt_weights: Prompt weight normalization set to {normalize}")

    def set_normalize_seed_weights(self, normalize: bool) -> None:
        """Set whether to normalize seed weights in blending operations."""
        self.normalize_seed_weights = normalize
        print(f"set_normalize_seed_weights: Seed weight normalization set to {normalize}")
        
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
        print(f"register_embedding_enhancer: Registered '{name}' enhancer with StreamParameterUpdater")
        
        # IMMEDIATELY apply enhancer to existing embeddings if they exist (fixes TensorRT timing issue)
        if hasattr(self.stream, 'prompt_embeds') and self.stream.prompt_embeds is not None:
            print(f"register_embedding_enhancer: Applying '{name}' enhancer to existing embeddings")
            print(f"register_embedding_enhancer: Current prompt_embeds shape: {self.stream.prompt_embeds.shape}")
            try:
                current_negative_embeds = getattr(self.stream, 'negative_prompt_embeds', None)
                enhanced_prompt_embeds, enhanced_negative_embeds = enhancer_func(
                    self.stream.prompt_embeds, current_negative_embeds
                )
                self.stream.prompt_embeds = enhanced_prompt_embeds
                if enhanced_negative_embeds is not None:
                    self.stream.negative_prompt_embeds = enhanced_negative_embeds
                print(f"register_embedding_enhancer: Enhanced prompt_embeds shape: {self.stream.prompt_embeds.shape}")
            except Exception as e:
                print(f"register_embedding_enhancer: Error applying '{name}' enhancer immediately: {e}")
                import traceback
                traceback.print_exc()
    
    def unregister_embedding_enhancer(self, enhancer_func) -> None:
        """Unregister a specific embedding enhancer function."""
        original_length = len(self._embedding_enhancers)
        self._embedding_enhancers = [(func, name) for func, name in self._embedding_enhancers if func != enhancer_func]
        removed_count = original_length - len(self._embedding_enhancers)
        if removed_count > 0:
            print(f"unregister_embedding_enhancer: Removed {removed_count} enhancer(s) from StreamParameterUpdater")
    
    def clear_embedding_enhancers(self) -> None:
        """Clear all embedding enhancers."""
        enhancer_count = len(self._embedding_enhancers)
        self._embedding_enhancers.clear()
        if enhancer_count > 0:
            print(f"clear_embedding_enhancers: Removed {enhancer_count} enhancer(s) from StreamParameterUpdater")

    def _normalize_weights(self, weights: List[float], normalize: bool) -> torch.Tensor:
        """Generic weight normalization helper"""
        weights_tensor = torch.tensor(weights, device=self.stream.device, dtype=self.stream.dtype)
        if normalize:
            weights_tensor = weights_tensor / weights_tensor.sum()
        return weights_tensor

    def _validate_index(self, index: int, item_list: List, operation_name: str) -> bool:
        """Generic index validation helper"""
        if not item_list:
            print(f"{operation_name}: Warning: No current item list")
            return False
            
        if index < 0 or index >= len(item_list):
            print(f"{operation_name}: Warning: Index {index} out of range (0-{len(item_list)-1})")
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
        seed_list: Optional[List[Tuple[int, float]]] = None,
        seed_interpolation_method: Literal["linear", "slerp"] = "linear",
        controlnet_strengths: Optional[List[float]] = None,
        ipadapter_strengths: Optional[List[float]] = None,
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
                print("update_stream_params: Warning: guidance_scale > 1.0 with cfg_type='none' will have no effect")
            self.stream.guidance_scale = guidance_scale
            
        if delta is not None:
            self.stream.delta = delta
            
        if seed is not None:
            self._update_seed(seed)
        
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
        
        # Handle ControlNet strength updates
        if controlnet_strengths is not None:
            self._update_controlnet_strengths(controlnet_strengths)
        
        # Handle IPAdapter strength updates
        if ipadapter_strengths is not None:
            self._update_ipadapter_strengths(ipadapter_strengths)
        
        if t_index_list is not None:
            self._recalculate_timestep_dependent_params(t_index_list)

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
            print("_apply_prompt_blending: Warning: No cached embeddings found")
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
            print(f"[DEBUG] StreamParameterUpdater._apply_prompt_blending: Applying {len(self._embedding_enhancers)} enhancer(s)")
            print(f"[DEBUG] StreamParameterUpdater._apply_prompt_blending: Input final_prompt_embeds shape: {final_prompt_embeds.shape}")
            for enhancer_func, enhancer_name in self._embedding_enhancers:
                try:
                    enhanced_prompt_embeds, enhanced_negative_embeds = enhancer_func(
                        final_prompt_embeds, final_negative_embeds
                    )
                    print(f"[DEBUG] StreamParameterUpdater._apply_prompt_blending: After '{enhancer_name}' enhancement: {enhanced_prompt_embeds.shape}")
                    final_prompt_embeds = enhanced_prompt_embeds
                    if enhanced_negative_embeds is not None:
                        final_negative_embeds = enhanced_negative_embeds
                except Exception as e:
                    print(f"_apply_prompt_blending: Error in enhancer '{enhancer_name}': {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print(f"[DEBUG] StreamParameterUpdater._apply_prompt_blending: No enhancers to apply")
        
        # Set final embeddings on stream
        print(f"[DEBUG] StreamParameterUpdater._apply_prompt_blending: Setting final_prompt_embeds shape: {final_prompt_embeds.shape}")
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
            print("_apply_seed_blending: Warning: No cached noise tensors found")
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
            c_skip, c_out = self.stream.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
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

    def _update_controlnet_strengths(self, strengths: List[float]) -> None:
        """Update ControlNet conditioning scales."""
        # Check if ControlNet is available
        if not hasattr(self.stream, 'controlnet_scales'):
            print("update_stream_params: Warning: ControlNet not available, ignoring controlnet_strengths")
            return
        
        # Validate strength count
        if len(strengths) != len(self.stream.controlnet_scales):
            print(f"update_stream_params: Warning: ControlNet strength count {len(strengths)} doesn't match ControlNet count {len(self.stream.controlnet_scales)}")
            return
        
        # Update ControlNet scales
        for i, strength in enumerate(strengths):
            self.stream.controlnet_scales[i] = strength
        
        print(f"update_stream_params: Updated ControlNet strengths: {strengths}")

    def _update_ipadapter_strengths(self, strengths: List[float]) -> None:
        """Update IPAdapter conditioning scales."""
        # Check if IPAdapter is available
        if not hasattr(self.stream, 'ipadapter') or self.stream.ipadapter is None:
            print("update_stream_params: Warning: IPAdapter not available, ignoring ipadapter_strengths")
            return
        
        # For now, only support single IPAdapter (use first strength)
        if len(strengths) > 0:
            strength = strengths[0]
            self.stream.ipadapter.set_scale(strength)
            print(f"update_stream_params: Updated IPAdapter strength: {strength}")
        
        if len(strengths) > 1:
            print(f"update_stream_params: Warning: Multiple IPAdapter strengths provided but only first one used: {strengths}")

    @torch.no_grad()
    def get_current_prompts(self) -> List[Tuple[str, float]]:
        """Get the current prompt list with weights."""
        return self._current_prompt_list.copy()

    @torch.no_grad()
    def get_current_seeds(self) -> List[Tuple[int, float]]:
        """Get the current seed list with weights."""
        return self._current_seed_list.copy() 