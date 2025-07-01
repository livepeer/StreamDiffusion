from typing import List, Optional
import torch


class StreamParameterUpdater:
    def __init__(self, stream_diffusion):
        self.stream = stream_diffusion
    
    @torch.no_grad()
    def update_stream_params(
        self,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        delta: Optional[float] = None,
        t_index_list: Optional[List[int]] = None,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        # TODO: Review if we can update width/height here.

        """Update streaming parameters efficiently in a single call."""
        
        # Handle width/height updates for dynamic resolution
        '''
        if width is not None or height is not None:
            self._update_resolution(width, height)
        '''
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
        
        if t_index_list is not None:
            self._recalculate_timestep_dependent_params(t_index_list)
    
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

    ''' TESTING
    def _update_resolution(self, width: Optional[int], height: Optional[int]) -> None:
        """Update stream resolution and regenerate resolution-dependent tensors."""
        # Use current dimensions if only one is provided
        new_width = width if width is not None else self.stream.width
        new_height = height if height is not None else self.stream.height
        
        # Validate resolution parameters
        if new_width % 64 != 0 or new_height % 64 != 0:
            raise ValueError(f"Resolution must be multiples of 64. Got {new_width}x{new_height}")
        
        if not (512 <= new_width <= 1024) or not (512 <= new_height <= 1024):
            raise ValueError(f"Resolution must be between 512-1024. Got {new_width}x{new_height}")
        
        # Check if resolution actually changed
        if new_width == self.stream.width and new_height == self.stream.height:
            print(f"update_stream_params: Resolution unchanged ({new_width}x{new_height}), skipping update")
            return
        
        print(f"update_stream_params: Updating resolution from {self.stream.width}x{self.stream.height} to {new_width}x{new_height}")
        
        # Update stream dimensions
        self.stream.width = new_width
        self.stream.height = new_height
        self.stream.latent_height = new_height // 8  # Assuming VAE scale factor of 8
        self.stream.latent_width = new_width // 8
        
        # Regenerate resolution-dependent tensors
        if hasattr(self.stream, 'generator') and self.stream.generator is not None:
            # Regenerate init_noise with new dimensions
            self.stream.init_noise = torch.randn(
                (self.stream.batch_size, 4, self.stream.latent_height, self.stream.latent_width),
                generator=self.stream.generator,
            ).to(device=self.stream.device, dtype=self.stream.dtype)
            
            # Reset stock_noise to match new init_noise
            self.stream.stock_noise = torch.zeros_like(self.stream.init_noise)
        
        # Update x_t_latent_buffer if it exists
        if hasattr(self.stream, 'x_t_latent_buffer') and self.stream.x_t_latent_buffer is not None:
            if self.stream.denoising_steps_num > 1:
                self.stream.x_t_latent_buffer = torch.zeros(
                    (
                        (self.stream.denoising_steps_num - 1) * self.stream.frame_bff_size,
                        4,
                        self.stream.latent_height,
                        self.stream.latent_width,
                    ),
                    dtype=self.stream.dtype,
                    device=self.stream.device,
                )
        
        # Notify ControlNet pipeline if present
        if hasattr(self.stream, 'controlnets') and self.stream.controlnets:
            print(f"update_stream_params: Warning - ControlNet resolution updates require pipeline recreation for TensorRT engines")
            print(f"update_stream_params: Consider using PyTorch mode or rebuilding TensorRT engines for {new_width}x{new_height}")
        
        print(f"update_stream_params: Resolution update completed to {new_width}x{new_height}") 
        '''