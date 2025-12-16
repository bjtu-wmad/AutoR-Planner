"""
KV Cache-enabled Causal Attention Module for Epona
Adapted from Self-Forcing's CausalWanSelfAttention to work with Epona's architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rope_2d import apply_rotary_emb


class CausalSelfAttentionWithCache(nn.Module):
    """
    Causal self-attention with KV caching support.
    
    This module extends Epona's CausalSpaceSelfAttention with KV caching
    mechanism from Self-Forcing, enabling efficient autoregressive generation.
    
    Args:
        config: Configuration object containing:
            - n_embd: embedding dimension
            - n_head: number of attention heads
            - attn_pdrop: attention dropout rate
            - resid_pdrop: residual dropout rate
            - token_size_dict: dictionary with token size information
            - patch_size: tuple of (height, width)
        local_attn_size: Size of local attention window (-1 for global attention)
        sink_size: Number of sink tokens to preserve when evicting from cache
    """
    
    def __init__(self, config, local_attn_size=-1, sink_size=0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        
        # Attention layers
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Dropout
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.attn_dropout_rate = config.attn_pdrop
        
        # Q, K normalization
        self.qk_norm = True
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(config.n_embd)
            self.k_norm = nn.LayerNorm(config.n_embd)
        else:
            self.q_norm = self.k_norm = nn.Identity()
        
        # Token configuration
        self.pose_tokens_num = config.token_size_dict['pose_tokens_size']
        self.img_tokens_num = config.token_size_dict['img_tokens_size']
        self.yaw_token_size = config.token_size_dict['yaw_token_size']
        self.total_tokens_num = config.token_size_dict['total_tokens_size']
        
        # RoPE configuration
        self.patch_size = config.patch_size
        from utils.rope_2d import compute_axial_cis
        self.freqs_cis_singlescale = compute_axial_cis(
            dim=config.n_embd // self.n_head,
            end_x=self.patch_size[0],
            end_y=self.patch_size[1],
            theta=1000.0
        )
        
        # Maximum attention size for caching
        if local_attn_size == -1:
            self.max_attention_size = 32760  # Global attention
        else:
            # Local attention window size
            self.max_attention_size = local_attn_size * self.total_tokens_num
    
    def forward(self, x, attn_mask=None, kv_cache=None, current_start=0):
        """
        Forward pass with optional KV caching.
        
        Args:
            x: Input tensor [B, T, C]
            attn_mask: Attention mask (used when kv_cache is None)
            kv_cache: Dictionary containing:
                - "k": cached keys [B, cache_size, n_head, head_dim]
                - "v": cached values [B, cache_size, n_head, head_dim]
                - "global_end_index": global position index
                - "local_end_index": local cache position index
            current_start: Starting position in the global sequence
        
        Returns:
            Output tensor [B, T, C]
        """
        B, T, C = x.size()
        
        # Compute Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, nh, T, hs]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, nh, T, hs]
        
        # Apply RoPE to image tokens only (not pose/yaw tokens)
        if T > self.pose_tokens_num + self.yaw_token_size:
            q_img = q[:, :, self.pose_tokens_num + self.yaw_token_size:, :]
            k_img = k[:, :, self.pose_tokens_num + self.yaw_token_size:, :]
            
            img_seq_len = T - self.pose_tokens_num - self.yaw_token_size
            q_img_rope, k_img_rope = apply_rotary_emb(
                q_img, k_img,
                freqs_cis=self.freqs_cis_singlescale[:img_seq_len]
            )
            
            q = torch.cat([
                q[:, :, :self.pose_tokens_num + self.yaw_token_size, :],
                q_img_rope
            ], dim=2)
            k = torch.cat([
                k[:, :, :self.pose_tokens_num + self.yaw_token_size, :],
                k_img_rope
            ], dim=2)
        
        # Different paths for cached vs non-cached attention
        if kv_cache is None:
            # Standard attention without caching
            if attn_mask is not None:
                attn_mask = attn_mask.to(q.dtype)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout_rate
            ).transpose(1, 2).contiguous().view(B, T, C)
        else:
            # Cached attention for autoregressive generation
            y = self._cached_attention(q, k, v, kv_cache, current_start, B, T, C)
        
        # Output projection
        y = self.resid_drop(self.proj(y))
        return y
    
    def _cached_attention(self, q, k, v, kv_cache, current_start, B, T, C):
        """
        Perform attention with KV caching.
        
        This implements the caching logic from Self-Forcing, managing
        a rolling window of cached key-value pairs.
        """
        current_end = current_start + T
        num_new_tokens = T
        
        # Get current cache state
        kv_cache_size = kv_cache["k"].shape[1]
        local_end_index = kv_cache["local_end_index"].item()
        global_end_index = kv_cache["global_end_index"].item()
        
        # Reshape k, v for caching [B, nh, T, hs] -> [B, T, nh, hs]
        k_cache = k.transpose(1, 2).contiguous()
        v_cache = v.transpose(1, 2).contiguous()
        
        # Check if we need to evict old tokens from cache
        if (self.local_attn_size != -1 and
            current_end > global_end_index and
            num_new_tokens + local_end_index > kv_cache_size):
            
            # Calculate eviction
            sink_tokens = self.sink_size * self.total_tokens_num
            num_evicted_tokens = num_new_tokens + local_end_index - kv_cache_size
            num_rolled_tokens = local_end_index - num_evicted_tokens - sink_tokens
            
            # Roll the cache (keep sink tokens, evict middle, keep recent)
            kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["k"][:, sink_tokens + num_evicted_tokens:
                             sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["v"][:, sink_tokens + num_evicted_tokens:
                             sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            
            # Update local index
            local_end_index = local_end_index + current_end - global_end_index - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
        else:
            # No eviction needed, just append
            local_end_index = local_end_index + current_end - global_end_index
            local_start_index = local_end_index - num_new_tokens
        
        # Store new keys and values in cache
        kv_cache["k"][:, local_start_index:local_end_index] = k_cache
        kv_cache["v"][:, local_start_index:local_end_index] = v_cache
        
        # Perform attention using cached K, V
        # Use only the relevant window from cache
        cache_start = max(0, local_end_index - self.max_attention_size)
        cached_k = kv_cache["k"][:, cache_start:local_end_index]
        cached_v = kv_cache["v"][:, cache_start:local_end_index]
        
        # Reshape back to [B, nh, seq_len, hs] for attention
        cached_k = cached_k.transpose(1, 2)
        cached_v = cached_v.transpose(1, 2)
        
        # Compute attention
        y = F.scaled_dot_product_attention(
            q, cached_k, cached_v,
            dropout_p=self.attn_dropout_rate if self.training else 0.0
        ).transpose(1, 2).contiguous().view(B, T, C)
        
        # Update cache indices
        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)
        
        return y


def initialize_kv_cache(num_blocks, batch_size, cache_size, n_heads, head_dim, dtype, device):
    """
    Initialize KV cache for multiple attention blocks.
    
    Args:
        num_blocks: Number of attention blocks
        batch_size: Batch size
        cache_size: Maximum cache size per block
        n_heads: Number of attention heads
        head_dim: Dimension per head
        dtype: Data type for cache tensors
        device: Device to create tensors on
    
    Returns:
        List of KV cache dictionaries, one per block
    """
    kv_cache_list = []
    for _ in range(num_blocks):
        kv_cache_list.append({
            "k": torch.zeros([batch_size, cache_size, n_heads, head_dim],
                           dtype=dtype, device=device),
            "v": torch.zeros([batch_size, cache_size, n_heads, head_dim],
                           dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
        })
    return kv_cache_list


def reset_kv_cache(kv_cache_list):
    """
    Reset all KV caches to initial state.
    
    Args:
        kv_cache_list: List of KV cache dictionaries
    """
    for kv_cache in kv_cache_list:
        kv_cache["global_end_index"].fill_(0)
        kv_cache["local_end_index"].fill_(0)
