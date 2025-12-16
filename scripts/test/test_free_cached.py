"""
Example script demonstrating how to use Epona with KV caching for inference.
This is a simplified version of test_free.py that uses the new caching pipeline.
"""

import os
import cv2
import sys
import time
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from einops import rearrange
from torch.utils.data import DataLoader, Subset

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
print("Root path:", root_path)
sys.path.append(root_path)

from utils.utils import *
from utils.testing_utils import create_mp4_imgs
from dataset.dataset_nuplan import NuPlan
from models.model import TrainTransformersDiT
from models.modules.tokenizer import VAETokenizer
from mmengine.config import Config
from utils.preprocess import get_rel_pose

# Import the new causal inference pipeline
from models.epona_causal_inference import EponaCausalInferencePipeline


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_video_path', type=str, default='test_videos')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--config', default='configs/dit_config_dcae_nuplan_cached.py', type=str)
    parser.add_argument('--resume_path', default=None, type=str, help='pretrained path')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--end_id', type=int, default=500)
    parser.add_argument('--use_kv_cache', action='store_true', 
                       help='Enable KV caching (recommended for speed)')
    
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.__dict__)
    return cfg


def test_with_kv_cache(val_data, model, args, tokenizer):
    """
    Test Epona with KV caching enabled for efficient long-video generation.
    """
    condition_frames = args.condition_frames
    save_path = os.path.join(args.save_video_path, args.exp_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize the causal inference pipeline
    pipeline = EponaCausalInferencePipeline(model, tokenizer, args)
    
    print(f"Using KV Cache: {args.use_kv_cache}")
    print(f"Local Attention Size: {args.local_attn_size}")
    
    with torch.no_grad():
        for i, (img, rot_matrix) in tqdm(enumerate(val_data)):
            video_save_path = os.path.join(save_path, f'cached_{args.start_id}')
            os.makedirs(video_save_path, exist_ok=True)
            
            model.eval()
            img = img.cuda()
            rot_matrix = rot_matrix.cuda()
            
            # Encode initial frames
            print("Encoding initial frames...")
            start_time = time.time()
            start_latents = tokenizer.encode_to_z(img[:, :condition_frames])
            print(f"Encoding time: {time.time() - start_time:.2f}s")
            
            # Save condition frames
            condition_imgs = []
            for j in range(condition_frames):
                img_pred = tokenizer.z_to_image(
                    rearrange(start_latents[:, j, ...], 'b (h w) c -> b h w c',
                             h=args.image_size[0]//(args.downsample_size*args.patch_size),
                             w=args.image_size[1]//(args.downsample_size*args.patch_size))
                )
                img_np = (img_pred[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')[:, :, ::-1]
                condition_imgs.append(img_np)
                cv2.imwrite(os.path.join(video_save_path, f'{j}.png'), img_np)
            
            # Generate video with KV caching
            print(f"Generating {args.test_video_frames} frames...")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                results = pipeline.generate_long_video(
                    start_latents=start_latents,
                    rot_matrix=rot_matrix,
                    num_frames=args.test_video_frames,
                    condition_frames=condition_frames,
                    save_intermediate=False
                )
            
            # Extract results
            all_latents = results['latents']
            all_trajectories = results['trajectories']
            timings = results['timings']
            
            # Print timing statistics
            pipeline.print_timing_stats(timings)
            
            # Decode latents to video frames
            print("Decoding latents to video...")
            generated_frames = []
            for frame_idx in range(condition_frames, all_latents.shape[1]):
                frame_latent = all_latents[:, frame_idx:frame_idx+1, :, :]
                frame_latent_reshaped = rearrange(frame_latent, 'b f (h w) c -> (b f) h w c',
                                                h=args.image_size[0]//(args.downsample_size*args.patch_size),
                                                w=args.image_size[1]//(args.downsample_size*args.patch_size))
                
                img_pred = tokenizer.z_to_image(frame_latent_reshaped, is_video=False)
                img_np = (img_pred[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')[:, :, ::-1]
                generated_frames.append(img_np)
                cv2.imwrite(os.path.join(video_save_path, f'{frame_idx}.png'), img_np)
            
            # Create output video
            all_frames = condition_imgs + generated_frames
            create_mp4_imgs(args, all_frames, video_save_path, fps=5)
            
            # Save trajectories
            if all_trajectories is not None:
                torch.save(all_trajectories, os.path.join(video_save_path, 'pred_traj.pt'))
                print(f"Saved trajectory shape: {all_trajectories.shape}")
            
            print(f"Video saved to: {video_save_path}")


def main(args):
    local_rank = 0
    
    # Load model
    print(f"Loading model from: {args.resume_path}")
    model = TrainTransformersDiT(
        args,
        load_path=args.resume_path,
        local_rank=local_rank,
        condition_frames=args.condition_frames
    )
    
    # Load dataset
    test_dataset = NuPlan(
        'nuplan-v1.1',
        'nuplan_meta',
        split='test',
        condition_frames=args.condition_frames + args.traj_len,
        downsample_fps=args.downsample_fps,
        h=args.image_size[0],
        w=args.image_size[1]
    )
    
    start_id, end_id = args.start_id, min(args.end_id, len(test_dataset))
    test_dataset = Subset(test_dataset, list(range(start_id, end_id)))
    
    print(f"Dataset length: {len(test_dataset)}, {start_id}-{end_id}")
    print(f"Condition frames: {args.condition_frames}")
    
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    tokenizer = VAETokenizer(args, local_rank)
    
    # Run inference
    test_with_kv_cache(test_dataloader, model, args, tokenizer)


if __name__ == "__main__":
    args = add_arguments()
    print(args)
    
    # Set random seeds
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    
    main(args)
