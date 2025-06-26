from typing import List, Optional, Dict, Tuple, Literal
import torch
import torch.nn.functional as F


class StreamParameterUpdater:
    def __init__(self, stream_diffusion):
        self.stream = stream_diffusion
        # Prompt blending caches
        self._prompt_cache: Dict[int, Dict] = {}
        self._current_prompt_list: List[Tuple[str, float]] = []
        self._current_negative_prompt: str = ""
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cache_info(self) -> Dict:
        """Get cache statistics for monitoring performance."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        return {
            "cached_prompts": len(self._prompt_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": f"{hit_rate:.2%}",
            "current_prompts": len(self._current_prompt_list)
        }
    
    def clear_caches(self) -> None:
        """Clear all caches to free memory."""
        self._prompt_cache.clear()
        self._current_prompt_list.clear()
        self._current_negative_prompt = ""
        self._cache_hits = 0
        self._cache_misses = 0

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
        interpolation_method: Literal["linear", "slerp"] = "slerp",
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
                interpolation_method=interpolation_method
            )
        
        if t_index_list is not None:
            self._recalculate_timestep_dependent_params(t_index_list)

    @torch.no_grad()
    def update_prompt_weights(
        self, 
        prompt_weights: List[float],
        interpolation_method: Literal["linear", "slerp"] = "slerp"
    ) -> None:
        """Update weights for current prompt list without re-encoding prompts."""
        if not self._current_prompt_list:
            print("update_prompt_weights: Warning: No current prompt list to update weights for")
            return
            
        if len(prompt_weights) != len(self._current_prompt_list):
            print(f"update_prompt_weights: Warning: Weight count {len(prompt_weights)} doesn't match prompt count {len(self._current_prompt_list)}")
            return
        
        # Update the current prompt list with new weights
        updated_prompt_list = []
        for i, (prompt_text, _) in enumerate(self._current_prompt_list):
            updated_prompt_list.append((prompt_text, prompt_weights[i]))
        
        self._current_prompt_list = updated_prompt_list
        
        # Recompute blended embeddings with new weights
        self._apply_prompt_blending(interpolation_method)

    @torch.no_grad()
    def _update_blended_prompts(
        self,
        prompt_list: List[Tuple[str, float]],
        negative_prompt: str = "",
        interpolation_method: Literal["linear", "slerp"] = "slerp"
    ) -> None:
        """Update prompt embeddings using multiple weighted prompts."""
        # Store current state
        self._current_prompt_list = prompt_list.copy()
        self._current_negative_prompt = negative_prompt
        
        # Encode any new prompts and cache them
        self._cache_prompt_embeddings(prompt_list, negative_prompt)
        
        # Apply blending
        self._apply_prompt_blending(interpolation_method)

    def _cache_prompt_embeddings(
        self, 
        prompt_list: List[Tuple[str, float]], 
        negative_prompt: str
    ) -> None:
        """Cache prompt embeddings for efficient reuse."""
        for idx, (prompt_text, weight) in enumerate(prompt_list):
            if idx not in self._prompt_cache or self._prompt_cache[idx]['text'] != prompt_text:
                # Cache miss - encode the prompt
                self._cache_misses += 1
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
                self._cache_hits += 1

    def _apply_prompt_blending(self, interpolation_method: Literal["linear", "slerp"]) -> None:
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
        weights = torch.tensor(weights, device=self.stream.device, dtype=self.stream.dtype)
        weights = weights / weights.sum()
        
        # Apply interpolation
        if interpolation_method == "slerp" and len(embeddings) == 2:
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
            self.stream.prompt_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)
        else:
            # No CFG, just use the blended embeddings
            self.stream.prompt_embeds = combined_embeds.repeat(self.stream.batch_size, 1, 1)

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
    def _update_blended_seeds(self, seed_list: List[Tuple[int, float]]) -> None:
        """Blend multiple seeds with weights to create diverse noise."""
        if not seed_list:
            return
            
        # Generate noise tensors for each seed
        noise_tensors = []
        weights = []
        
        for seed_value, weight in seed_list:
            generator = torch.Generator(device=self.stream.device)
            generator.manual_seed(seed_value)
            
            noise = torch.randn(
                (self.stream.batch_size, 4, self.stream.latent_height, self.stream.latent_width),
                generator=generator,
                device=self.stream.device,
                dtype=self.stream.dtype
            )
            noise_tensors.append(noise)
            weights.append(weight)
        
        # Normalize weights
        weights = torch.tensor(weights, device=self.stream.device, dtype=self.stream.dtype)
        weights = weights / weights.sum()
        
        # Blend noise tensors
        blended_noise = torch.zeros_like(noise_tensors[0])
        for noise, weight in zip(noise_tensors, weights):
            blended_noise += weight * noise
        
        # Update stream noise
        self.stream.init_noise = blended_noise
        self.stream.stock_noise = torch.zeros_like(self.stream.init_noise)

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

    @torch.no_grad()
    def update_prompt_at_index(
        self, 
        index: int, 
        new_prompt: str,
        interpolation_method: Literal["linear", "slerp"] = "slerp"
    ) -> None:
        """Update a single prompt at the specified index without re-encoding others."""
        if not self._current_prompt_list:
            print("update_prompt_at_index: Warning: No current prompt list")
            return
            
        if index < 0 or index >= len(self._current_prompt_list):
            print(f"update_prompt_at_index: Warning: Index {index} out of range (0-{len(self._current_prompt_list)-1})")
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
                self._cache_hits += 1
            else:
                # Encode new prompt
                self._cache_misses += 1
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
        self._apply_prompt_blending(interpolation_method)

    @torch.no_grad()
    def get_current_prompts(self) -> List[Tuple[str, float]]:
        """Get the current prompt list with weights."""
        return self._current_prompt_list.copy()

    @torch.no_grad()
    def add_prompt(
        self, 
        prompt: str, 
        weight: float = 1.0,
        interpolation_method: Literal["linear", "slerp"] = "slerp"
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
        self._cache_misses += 1
        
        # Recompute blended embeddings
        self._apply_prompt_blending(interpolation_method)

    @torch.no_grad()
    def remove_prompt_at_index(
        self, 
        index: int,
        interpolation_method: Literal["linear", "slerp"] = "slerp"
    ) -> None:
        """Remove a prompt at the specified index."""
        if not self._current_prompt_list:
            print("remove_prompt_at_index: Warning: No current prompt list")
            return
            
        if index < 0 or index >= len(self._current_prompt_list):
            print(f"remove_prompt_at_index: Warning: Index {index} out of range")
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
        new_cache = {}
        for cache_idx, cache_data in self._prompt_cache.items():
            if cache_idx < index:
                new_cache[cache_idx] = cache_data
            elif cache_idx > index:
                new_cache[cache_idx - 1] = cache_data
        self._prompt_cache = new_cache
        
        # Recompute blended embeddings
        self._apply_prompt_blending(interpolation_method) 