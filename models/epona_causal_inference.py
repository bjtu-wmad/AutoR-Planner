"""
Epona Causal Inference Pipeline with KV Caching
Enables efficient long-video generation through cached autoregressive inference
"""

import torch
import time
from einops import rearrange


class EponaCausalInferencePipeline:
    """
    Inference pipeline for Epona with KV caching support.
    
    This pipeline manages the autoregressive generation process with KV caching,
    coordinating between trajectory prediction (TrajDiT), feature encoding (STT),
    and visual generation (FluxDiT).
    
    Args:
        model: TrainTransformersDiT model
        tokenizer: VAE tokenizer for encoding/decoding
        args: Configuration arguments
    """
    
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        
        # KV cache configuration
        self.local_attn_size = getattr(args, 'local_attn_size', -1)
        self.use_kv_cache = getattr(args, 'use_kv_cache', True)
        
        # Cache state
        self.kv_cache = None
        
    def reset_cache(self):
        """Reset the KV cache to initial state."""
        if self.kv_cache is not None:
            from models.modules.kv_cache_attention import reset_kv_cache
            reset_kv_cache(self.kv_cache)
        self.kv_cache = None
    
    @torch.no_grad()
    def generate_long_video(
        self,
        start_latents,
        rot_matrix,
        num_frames=60,
        condition_frames=3,
        save_intermediate=False
    ):
        """
        Generate long video using KV caching for efficiency.
        
        Args:
            start_latents: Initial condition frames [B, F, L, C]
            rot_matrix: Full rotation matrix for trajectory [B, T, ...]
            num_frames: Total number of frames to generate
            condition_frames: Number of conditioning frames
            save_intermediate: Whether to save intermediate results
        
        Returns:
            Dictionary containing:
                - latents: Generated latent sequences [B, num_frames, L, C]
                - trajectories: Predicted trajectories
                - timings: Performance timing information
        """
        B = start_latents.shape[0]
        device = start_latents.device
        
        # Initialize storage
        all_latents = [start_latents]
        all_trajectories = []
        timings = {
            'stt': [],
            'traj_dit': [],
            'vis_dit': [],
            'total': 0
        }
        
        # Reset cache for new sequence
        self.reset_cache()
        
        # Get initial pose and yaw from rotation matrix
        from utils.preprocess import get_rel_pose
        pose_total, yaw_total = get_rel_pose(rot_matrix)
        pose = pose_total[:, :condition_frames+1, ...]
        yaw = yaw_total[:, :condition_frames+1, ...]
        
        start_time = time.time()
        
        # Autoregressive generation loop
        current_latents = start_latents
        for frame_idx in range(num_frames):
            print(f"Generating frame {frame_idx}/{num_frames}")
            
            # Phase 1: STT encoding with KV cache
            stt_start = time.time()
            if self.use_kv_cache:
                stt_features, pose_emb, self.kv_cache = self.model.model.evaluate_with_cache(
                    current_latents,
                    pose,
                    yaw,
                    kv_cache=self.kv_cache,
                    current_frame=frame_idx,
                    local_attn_size=self.local_attn_size,
                    sample_last=True
                )
            else:
                stt_features, pose_emb = self.model.model.evaluate(
                    current_latents,
                    pose,
                    yaw,
                    sample_last=True
                )
            stt_time = time.time() - stt_start
            timings['stt'].append(stt_time)
            
            # Phase 2: Trajectory prediction
            traj_start = time.time()
            predict_traj = self._predict_trajectory(stt_features)
            traj_time = time.time() - traj_start
            timings['traj_dit'].append(traj_time)
            all_trajectories.append(predict_traj)
            
            # Update pose and yaw from predicted trajectory
            predict_pose = predict_traj[:, 0:1, 0:2]
            predict_yaw = predict_traj[:, 0:1, 2:3]
            
            # Phase 3: Visual generation
            vis_start = time.time()
            predict_latents = self._generate_visual(stt_features, predict_pose, predict_yaw)
            vis_time = time.time() - vis_start
            timings['vis_dit'].append(vis_time)
            
            # Store results
            all_latents.append(predict_latents)
            
            # Update sliding window
            predict_latents_reshaped = rearrange(predict_latents, '(b f) h w c -> b f (h w) c', f=1)
            current_latents = torch.cat(
                (current_latents[:, 1:condition_frames, ...], predict_latents_reshaped),
                dim=1
            )
            
            # Update pose/yaw window
            pose = torch.cat((pose[:, 1:condition_frames, ...], predict_pose, predict_pose), dim=1)
            yaw = torch.cat((yaw[:, 1:condition_frames, ...], predict_yaw, predict_yaw), dim=1)
        
        timings['total'] = time.time() - start_time
        
        # Concatenate all results
        all_latents_concat = torch.cat(all_latents, dim=1)
        all_trajectories_concat = torch.cat(all_trajectories) if all_trajectories else None
        
        return {
            'latents': all_latents_concat,
            'trajectories': all_trajectories_concat,
            'timings': timings
        }
    
    def _predict_trajectory(self, stt_features):
        """
        Predict trajectory using TrajDiT.
        
        Args:
            stt_features: Features from STT [B, L, C]
        
        Returns:
            Predicted trajectory [B, traj_len, 3]
        """
        from models.modules.sampling import prepare_ids, get_schedule
        
        bsz = stt_features.shape[0]
        h, w = self.model.h, self.model.w
        traj_len = self.model.traj_len
        
        _, _, traj_ids = prepare_ids(bsz, h, w, self.model.total_token_size, traj_len)
        _, cond_ids, _ = prepare_ids(bsz, h, w, self.model.total_token_size, traj_len)
        
        # Sample noise for trajectory
        noise_traj = torch.randn(bsz, traj_len, self.model.traj_token_size).to(stt_features)
        timesteps_traj = get_schedule(
            int(self.args.num_sampling_steps),
            traj_len
        )
        
        # Denormalize prediction
        predict_traj = self.model.traj_dit.sample(
            noise_traj, traj_ids, stt_features, cond_ids, timesteps_traj
        )
        predict_traj = self.model.denormalize_traj(predict_traj)
        
        return predict_traj
    
    def _generate_visual(self, stt_features, predict_pose, predict_yaw):
        """
        Generate visual content using FluxDiT.
        
        Args:
            stt_features: Features from STT [B, L, C]
            predict_pose: Predicted pose [B, 1, 2]
            predict_yaw: Predicted yaw [B, 1, 1]
        
        Returns:
            Generated latents [B, h, w, C]
        """
        from models.modules.tokenizer import poses_to_indices, yaws_to_indices
        from models.modules.sampling import prepare_ids, get_schedule
        
        bsz = stt_features.shape[0]
        h, w = self.model.h, self.model.w
        
        # Get pose embeddings
        predict_pose_idx = poses_to_indices(
            predict_pose,
            self.model.pose_x_vocab_size,
            self.model.pose_y_vocab_size
        )
        predict_yaw_idx = yaws_to_indices(predict_yaw, self.model.yaw_vocab_size)
        pose_emb = self.model.model.get_pose_emb(predict_pose_idx, predict_yaw_idx)
        
        # Prepare IDs
        img_ids, cond_ids, _ = prepare_ids(
            bsz, h, w, self.model.total_token_size, self.model.traj_len
        )
        
        # Sample noise for visual content
        noise = torch.randn(bsz, self.model.img_token_size, self.model.vae_emb_dim).to(stt_features)
        timesteps = get_schedule(int(self.args.num_sampling_steps), self.model.img_token_size)
        
        # Generate latents
        predict_latents = self.model.dit.sample(
            noise, img_ids, stt_features, cond_ids, pose_emb, timesteps
        )
        predict_latents = rearrange(predict_latents, 'b (h w) c -> b h w c', h=h, w=w)
        
        return predict_latents
    
    def decode_to_video(self, latents, fps=5):
        """
        Decode latents to video frames.
        
        Args:
            latents: Latent representations [B, F, h, w, C]
            fps: Frames per second for video
        
        Returns:
            Video frames as numpy arrays
        """
        B, F = latents.shape[:2]
        frames = []
        
        for frame_idx in range(F):
            frame_latent = latents[:, frame_idx:frame_idx+1, :, :]
            frame_latent_reshaped = rearrange(frame_latent, 'b f h w c -> (b f) h w c')
            
            # Decode frame
            frame_img = self.tokenizer.z_to_image(frame_latent_reshaped, is_video=False)
            frame_np = (frame_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')[:, :, ::-1]
            frames.append(frame_np)
        
        return frames
    
    def print_timing_stats(self, timings):
        """Print timing statistics for performance analysis."""
        print("\n" + "="*50)
        print("Performance Timing Statistics")
        print("="*50)
        
        if timings['stt']:
            print(f"STT Encoding:  {sum(timings['stt']):.2f}s "
                  f"(avg: {sum(timings['stt'])/len(timings['stt']):.3f}s/frame)")
        if timings['traj_dit']:
            print(f"Trajectory:    {sum(timings['traj_dit']):.2f}s "
                  f"(avg: {sum(timings['traj_dit'])/len(timings['traj_dit']):.3f}s/frame)")
        if timings['vis_dit']:
            print(f"Visual Gen:    {sum(timings['vis_dit']):.2f}s "
                  f"(avg: {sum(timings['vis_dit'])/len(timings['vis_dit']):.3f}s/frame)")
        
        print(f"Total Time:    {timings['total']:.2f}s")
        
        if timings['stt']:
            avg_per_frame = timings['total'] / len(timings['stt'])
            fps = 1.0 / avg_per_frame
            print(f"Average FPS:   {fps:.2f}")
        
        print("="*50 + "\n")
