import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class DistanceNorm(nn.Module):
    """Batch-level distance normalization."""
    def __init__(self, momentum: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, dist_w: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Normalize distance weights into a stable numeric range."""
        B, L, _ = dist_w.shape

        # Use log transform to improve numerical stability
        u = torch.log(dist_w.clamp_min(self.eps))

        if self.training:
            # Compute statistics only on valid positions
            if mask is not None:
                # Build a 2D mask: (B, L, L)
                mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, L, L)
                valid_mask = mask_2d.bool()

                if valid_mask.sum() > 1:  # need at least two elements to compute variance
                    valid_u = u[valid_mask]
                    batch_mean = valid_u.mean()
                    batch_var = valid_u.var(unbiased=False)
                else:
                    batch_mean = u.mean()
                    batch_var = u.var(unbiased=False)
            else:
                batch_mean = u.mean()
                batch_var = u.var(unbiased=False)

            # Update running statistics
            self.num_batches_tracked += 1
            if self.num_batches_tracked == 1:
                self.running_mean.copy_(batch_mean)
                self.running_var.copy_(batch_var)
            else:
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean, alpha=self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(batch_var, alpha=self.momentum)

            # Use current batch stats during training
            mean = batch_mean
            var = batch_var
        else:
            # Use running stats during inference
            mean = self.running_mean
            var = self.running_var

        # Standardize
        z = (u - mean) / torch.sqrt(var + self.eps)

        return z


class Adaptive_Dynamic_Threshold_Gate(nn.Module):
    """Learnable-threshold edge selection with sigmoid gating (multi-head soft gating)."""
    def __init__(
        self,
        num_heads: int = 1,
        init_threshold: float = 0.0,
        init_temperature: float = 5.0,
        target_sparsity: float = 0.2,
        sparsity_weight: float = 1e-4,
        min_gate_value: float = 1e-6
    ):
        super().__init__()
        self.num_heads = num_heads
        self.target_sparsity = target_sparsity
        self.sparsity_weight = sparsity_weight
        self.min_gate_value = min_gate_value

        # Learnable threshold per head
        self.threshold = nn.Parameter(torch.full((num_heads,), init_threshold))

        self.register_buffer('temperature', torch.full((num_heads,), init_temperature))
        # Distance bias (learnable)
        self.bias_weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, normalized_dist: torch.Tensor, raw_dist: torch.Tensor,
                training: bool = True, mask: Optional[torch.Tensor] = None):
        """Return multi-head soft gates and optional sparsity regularization."""
        B, L, _ = normalized_dist.shape
        H = self.num_heads

        # Expand to multi-head: (B, H, L, L)
        z_expanded = normalized_dist.unsqueeze(1).expand(B, H, L, L)

        # Align parameter shapes: (1, H, 1, 1)
        tau = self.threshold.view(1, H, 1, 1)
        beta = self.temperature.view(1, H, 1, 1)

        # Compute gates
        logits = beta * (z_expanded - tau)

        logits = torch.clamp(logits, min=-10, max=10)
        soft_gates = torch.sigmoid(logits)

        # Keep a lower bound
        soft_gates = torch.clamp(soft_gates, min=self.min_gate_value, max=1.0)
        bias_term = torch.exp(self.bias_weight * torch.log(raw_dist.clamp_min(1e-6)))
        soft_gates = soft_gates * bias_term.unsqueeze(1)  # (B,1,L,L) broadcast to heads

        sparsity_loss = None

        # Sparsity constraint
        if training and self.sparsity_weight > 0:
            if mask is not None:
                # Valid-position mask: (B, L, L)
                mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, L, L)
                # Expand to heads: (B, H, L, L)
                mask_expanded = mask_2d.unsqueeze(1).expand(B, H, L, L)
                valid = mask_expanded.bool()

                if valid.sum() > 0:
                    mean_sparsity = soft_gates[valid].mean()
                else:
                    mean_sparsity = torch.tensor(0.0, device=soft_gates.device)
            else:
                mean_sparsity = soft_gates.mean()

            sparsity_loss = self.sparsity_weight * torch.abs(mean_sparsity - self.target_sparsity)

        return soft_gates, sparsity_loss

class SparseSelfAttention(nn.Module):
    """Sparse self-attention."""
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        num_neighbors: int = 30,
        dropout: float = 0.1,
        # Neighbor selection method
        selection_method: str = "knn",  # "knn" or "learnable_threshold"
        # Learnable-threshold parameters
        threshold_target_sparsity: float = 0.2,
        threshold_sparsity_weight: float = 1e-4,
        distance_eps: float = 1e-6,
    ):
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.num_neighbors = num_neighbors
        self.selection_method = selection_method
        self.distance_eps = distance_eps

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(input_dim)

        # Attention projections
        self.W_Q = nn.Linear(input_dim, input_dim)
        self.W_K = nn.Linear(input_dim, input_dim)
        self.W_V = nn.Linear(input_dim, input_dim)

        # Distance bias parameter
        self.distance_bias_weight = nn.Parameter(torch.ones(1))

        # Neighbor selection
        if selection_method == "knn":
            self.knn_selector = None
        elif selection_method == "learnable_threshold":
            self.threshold_gating = Adaptive_Dynamic_Threshold_Gate(
                num_heads=num_heads,
                target_sparsity=threshold_target_sparsity,
                sparsity_weight=threshold_sparsity_weight
            )
        else:
            raise ValueError(f"Unknown selection_method: {selection_method}")


    def _apply_knn_selection(self, attention_scores: torch.Tensor, distance_weights: torch.Tensor) -> torch.Tensor:
        """KNN-based neighbor selection."""
        B, H, L, _ = attention_scores.shape

        # Determine k
        if self.knn_selector is not None:
            k = self.knn_selector.get_k(L)
        else:
            k = min(self.num_neighbors, L)

        if k <= 0 or k >= L:
            return attention_scores

        # Filter with distance weights
        if distance_weights.dim() == 4:
            distance_weights = distance_weights.squeeze(1)  # (B, L, L)

        # Top-k
        _, top_indices = torch.topk(distance_weights, k, dim=-1)  # (B, L, k)

        # Build mask
        mask = torch.zeros_like(distance_weights, dtype=torch.bool)  # (B, L, L)
        mask.scatter_(2, top_indices, True)

        # Apply mask to attention scores
        mask_expanded = mask.unsqueeze(1).expand_as(attention_scores)  # (B, H, L, L)
        attention_scores = attention_scores.masked_fill(~mask_expanded, float('-inf'))

        return attention_scores

    def _apply_threshold_gating(self, attention_scores, normalized_dist, raw_dist,
                                training: bool, padding_mask: Optional[torch.Tensor]):
        gates, sparsity_loss = self.threshold_gating(normalized_dist, raw_dist,
                                                    training=training, mask=padding_mask)
        gate_bias = torch.log(gates.clamp_min(self.distance_eps))
        attention_scores = attention_scores + gate_bias
        return attention_scores, sparsity_loss

    def forward(
        self,
        x: torch.Tensor,  # (B, L, input_dim)
        attention_mask: Optional[torch.Tensor] = None,  # (B, 1, 1, L)
        distance_weights: Optional[torch.Tensor] = None,  # (B, L, L)
        normalized_dist: Optional[torch.Tensor] = None,  # (B, L, L)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, _ = x.shape
        sparsity_loss = None

        # Build key padding mask
        key_pad = None
        padding_mask = None  # mask used for sparsity computation
        if attention_mask is not None:
            key_pad = (attention_mask[:, 0, 0, :] < 0)
            padding_mask = ~key_pad

        # Standard self-attention
        Q = self.W_Q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, Dh)
        K = self.W_K(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (B, H, L, L)

        # Distance bias
        if distance_weights is not None:
            if distance_weights.dim() == 3:
                distance_weights = distance_weights.unsqueeze(1)  # (B, 1, L, L)

            distance_bias = self.distance_bias_weight * torch.log(distance_weights.clamp_min(self.distance_eps))
            attention_scores = attention_scores + distance_bias

        # Neighbor selection
        if self.selection_method == "knn" and distance_weights is not None:
            attention_scores = self._apply_knn_selection(attention_scores, distance_weights)
        elif self.selection_method == "learnable_threshold" and normalized_dist is not None:
            raw_dist = distance_weights
            if raw_dist is not None and raw_dist.dim() == 4:
                raw_dist = raw_dist.squeeze(1)
            attention_scores, sparsity_loss = self._apply_threshold_gating(
                attention_scores, normalized_dist, raw_dist, self.training, padding_mask
            )

        # Apply mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Softmax + dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Attention-weighted sum
        output = torch.matmul(attention_probs, V)  # (B, H, L, Dh)

        # Merge heads + residual + layer norm
        output = output.transpose(1, 2).contiguous().view(B, L, self.input_dim)  # (B, L, input_dim)
        output = self.layernorm(output + x)

        return output, sparsity_loss

class GaussianRBFDistanceEncoder(nn.Module):
    """Gaussian RBF-based distance encoder."""
    def __init__(self,
                 num_kernels: int = 8,
                 distance_min: float = 0.0,
                 distance_max: float = 20.0,
                 sigma_factor: float = 1.0,
                 learnable_sigma: bool = False):
        super().__init__()
        self.num_kernels = num_kernels

        # Uniformly spaced centers in the distance range
        centers = torch.linspace(distance_min, distance_max, num_kernels)
        self.register_buffer('centers', centers)

        # sigma is half of the spacing between adjacent centers
        sigma_init = (distance_max - distance_min) / (num_kernels - 1) * sigma_factor

        if learnable_sigma:
            # Independent learnable sigma for each kernel
            self.log_sigma = nn.Parameter(torch.full((num_kernels,), math.log(sigma_init)))
        else:
            # Fixed sigma
            self.register_buffer('sigma', torch.tensor(sigma_init))
            self.log_sigma = None

    def forward(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        """Input distance matrix; output aggregated RBF feature."""
        B, L, _ = distance_matrix.shape

        # Expand dimensions
        dist_expanded = distance_matrix.unsqueeze(-1)  # (B, L, L, 1)
        centers_expanded = self.centers.view(1, 1, 1, -1)  # (1, 1, 1, num_kernels)

        # Differences to centers
        diff = dist_expanded - centers_expanded  # (B, L, L, num_kernels)

        # Compute RBF
        if self.log_sigma is not None:
            # Learnable sigma
            sigma = torch.exp(self.log_sigma).view(1, 1, 1, -1)  # (1, 1, 1, num_kernels)
        else:
            sigma = self.sigma

        rbf = torch.exp(-0.5 * (diff / sigma) ** 2)  # (B, L, L, num_kernels)

        weights = torch.softmax(-torch.abs(diff) / sigma, dim=-1)  # (B, L, L, num_kernels)
        rbf_aggregated = (rbf * weights).sum(dim=-1)  # (B, L, L)

        return rbf_aggregated


class MultiScaleRBFEncoder(nn.Module):
    """Multi-scale RBF encoder for short-, medium-, and long-range interactions."""
    def __init__(self,
                 short_min: float = 0.0, short_max: float = 8.0, short_kernels: int = 6,
                 medium_min: float = 5.0, medium_max: float = 15.0, medium_kernels: int = 8,
                 long_min: float = 10.0, long_max: float = 25.0, long_kernels: int = 6,
                 sigma_factor: float = 1.0, learnable_sigma: bool = False):
        super().__init__()

        # Short-range interactions
        self.short_range = GaussianRBFDistanceEncoder(
            num_kernels=short_kernels,
            distance_min=short_min,
            distance_max=short_max,
            sigma_factor=sigma_factor,
            learnable_sigma=learnable_sigma
        )

        # Medium-range interactions
        self.medium_range = GaussianRBFDistanceEncoder(
            num_kernels=medium_kernels,
            distance_min=medium_min,
            distance_max=medium_max,
            sigma_factor=sigma_factor,
            learnable_sigma=learnable_sigma
        )

        # Long-range interactions
        self.long_range = GaussianRBFDistanceEncoder(
            num_kernels=long_kernels,
            distance_min=long_min,
            distance_max=long_max,
            sigma_factor=sigma_factor,
            learnable_sigma=learnable_sigma
        )

        # Learnable fusion weights across scales
        self.fusion_weights = nn.Parameter(torch.ones(3))

    def forward(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        # Compute RBF features at each scale
        short_rbf = self.short_range(distance_matrix)
        medium_rbf = self.medium_range(distance_matrix)
        long_rbf = self.long_range(distance_matrix)

        # Normalize fusion weights
        weights = torch.softmax(self.fusion_weights, dim=0)

        # Weighted fusion
        combined = (weights[0] * short_rbf +
                    weights[1] * medium_rbf +
                    weights[2] * long_rbf)

        return combined


class RGT(nn.Module):
    """RBF-Gate Transformer."""
    def __init__(
        self,
        protein_in_dim: int,
        protein_out_dim: int = 64,
        target_dim: int = 1,
        fc_layer_num: int = 2,
        atten_layer_num: int = 2,
        atten_head: int = 4,
        num_neighbor: int = 30,
        drop_rate1: float = 0.2,
        drop_rate2: float = 0.0,

        # RBF distance encoding
        use_rbf_distance: bool = False,
        rbf_learnable_sigma: bool = False,
        rbf_sigma_factor: float = 1.0,
        rbf_short_min: float = 0.0,
        rbf_short_max: float = 8.0,
        rbf_short_kernels: int = 6,
        rbf_medium_min: float = 5.0,
        rbf_medium_max: float = 15.0,
        rbf_medium_kernels: int = 8,
        rbf_long_min: float = 10.0,
        rbf_long_max: float = 25.0,
        rbf_long_kernels: int = 6,
        rbf_scale_init: float = 0.1,

        # Neighbor selection method
        selection_method: str = "knn",  # "knn" or "learnable_threshold"
        threshold_target_sparsity: float = 0.2,
        threshold_sparsity_weight: float = 1e-4,

        cfg: Optional[Dict] = None,
        distance_eps: float = 1e-6,
    ):
        super().__init__()
        self.use_rbf_distance = use_rbf_distance
        self.atten_layer_num = atten_layer_num
        self.selection_method = selection_method
        self.cfg = cfg or {}
        self.distance_eps = distance_eps
        self._log_counter = 0

        # Input processing
        self.input_block = nn.Sequential(
            nn.LayerNorm(protein_in_dim, elementwise_affine=True),
            nn.Linear(protein_in_dim, protein_out_dim),
            nn.LeakyReLU(),
        )

        # Feed-forward blocks
        hidden = []
        for h in range(fc_layer_num - 1):
            blocks = [nn.LayerNorm(protein_out_dim, elementwise_affine=True), nn.Dropout(drop_rate1),
                     nn.Linear(protein_out_dim, protein_out_dim), nn.LeakyReLU()]
            if h == fc_layer_num - 2:
                blocks += [nn.LayerNorm(protein_out_dim, elementwise_affine=True)]
            hidden.extend(blocks)
        self.hidden_block = nn.Sequential(*hidden) if hidden else nn.Identity()

        # RBF encoder
        if use_rbf_distance:
            self.rbf_encoder = MultiScaleRBFEncoder(
                short_min=rbf_short_min, short_max=rbf_short_max, short_kernels=rbf_short_kernels,
                medium_min=rbf_medium_min, medium_max=rbf_medium_max, medium_kernels=rbf_medium_kernels,
                long_min=rbf_long_min, long_max=rbf_long_max, long_kernels=rbf_long_kernels,
                sigma_factor=rbf_sigma_factor, learnable_sigma=rbf_learnable_sigma
            )
            self.rbf_scale = nn.Parameter(torch.tensor(rbf_scale_init))  # learnable fusion scale

        # Distance normalization (for the threshold-based method)
        if selection_method == "learnable_threshold":
            self.dist_norm = DistanceNorm(eps=distance_eps)

        # Attention layers
        self.layers = nn.ModuleList([
            nn.ModuleList([
                SparseSelfAttention(
                    protein_out_dim,
                    atten_head,
                    num_neighbors=num_neighbor,
                    dropout=drop_rate2,
                    selection_method=selection_method,
                    threshold_target_sparsity=threshold_target_sparsity,
                    threshold_sparsity_weight=threshold_sparsity_weight,
                    distance_eps=distance_eps,
                ),
                FeedForward(protein_out_dim, protein_out_dim * 4, drop_rate2),
            ])
            for i in range(atten_layer_num)
        ])

        # Output projection
        self.logit = nn.Linear(protein_out_dim, target_dim)

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def compute_distance_weights(self, protein_dist_matrix: torch.Tensor, protein_masks: torch.Tensor) -> torch.Tensor:
        """Compute distance weights, optionally enhanced by RBF (if enabled)."""
        # Base analytic distance weights
        analytic = 1.0 / torch.sqrt(1.0 + protein_dist_matrix.clamp_min(0.0))

        # If RBF is enabled, enhance weights
        if self.use_rbf_distance and hasattr(self, 'rbf_encoder'):
            rbf_features = self.rbf_encoder(protein_dist_matrix)  # (B, L, L)

            # Fusion strategy: multiplicative enhancement (preserve analytic decay)
            dist_w = analytic * (1.0 + self.rbf_scale * rbf_features)
            # Alternatively: additive enhancement
            # dist_w = analytic + self.rbf_scale * rbf_features
        else:
            dist_w = analytic

        # Apply mask
        mask_2d = protein_masks.unsqueeze(1) * protein_masks.unsqueeze(2)  # (B, L, L)
        dist_w = dist_w * mask_2d

        # Clamp to a stable range
        dist_w = dist_w.clamp(self.distance_eps, 1.0)
        return dist_w

    def forward(
        self,
        protein_node_features: torch.Tensor,  # (B, L, F)
        protein_edge_features: Optional[torch.Tensor],
        protein_dist_matrix: torch.Tensor,    # (B, L, L)
        protein_masks: torch.Tensor,          # (B, L) 1/0
    ) -> Dict[str, torch.Tensor]:
        # Input processing
        x = self.input_block(protein_node_features)
        x = self.hidden_block(x)

        B, L, _ = x.shape

        # Compute distance weights
        dist_w = self.compute_distance_weights(protein_dist_matrix, protein_masks)

        # Distance normalization
        normalized_dist = None
        if self.selection_method == "learnable_threshold":
            normalized_dist = self.dist_norm(dist_w, protein_masks)

        # Attention mask
        add_mask = (1.0 - protein_masks) * -10000.0  # (B, L)
        add_mask = add_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)

        # Collect regularization loss
        total_sparsity_loss = None

        # Attention + FFN blocks
        for layer_idx, (attn, ffn) in enumerate(self.layers):
            x, sparsity_loss = attn(
                x,
                attention_mask=add_mask,
                distance_weights=dist_w,
                normalized_dist=normalized_dist,
            )

            # Accumulate sparsity term
            if sparsity_loss is not None:
                total_sparsity_loss = sparsity_loss if total_sparsity_loss is None else (total_sparsity_loss + sparsity_loss)

            # Feed-forward network (residual)
            x = x + ffn(x)

        # Final logits
        logits = self.logit(x).squeeze(-1)

        # Pack outputs
        output_dict = {'logits': logits}

        # Sparsity term summary
        if total_sparsity_loss is not None and total_sparsity_loss.item() > 0:
            output_dict['sparsity_loss'] = total_sparsity_loss
        return output_dict

    def compute_loss(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor, masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute total loss including optional regularization terms."""
        logits = outputs['logits']

        # Main classification loss
        masks_float = masks.float()
        loss_all = self.criterion(logits, labels)  # (B, L)
        loss_masked = loss_all * masks_float
        main_loss = loss_masked.sum() / masks_float.sum()

        loss_dict = {'main_loss': main_loss}
        total_loss = main_loss

        # Sparsity regularization
        if 'sparsity_loss' in outputs:
            sparsity_loss = outputs['sparsity_loss']
            loss_dict['sparsity_loss'] = sparsity_loss
            total_loss += sparsity_loss

        loss_dict['total_loss'] = total_loss
        return loss_dict

def create_rgt_model(device: torch.device, cfg: Dict) -> RGT:

    model = RGT(
        # Base architecture
        protein_in_dim=cfg["protein_in_dim"],
        protein_out_dim=cfg.get("hidden_unit", 64),
        target_dim=cfg.get("class_num", 1),
        fc_layer_num=cfg.get("fc_layer", 2),
        atten_layer_num=cfg.get("self_atten_layer", 2),
        atten_head=cfg.get("attention_heads", 4),
        num_neighbor=cfg.get("num_neighbor", 30),
        drop_rate1=cfg.get("fc_dropout", 0.2),
        drop_rate2=cfg.get("attention_dropout", 0.0),

        # RBF parameters
        use_rbf_distance=cfg.get("use_rbf_distance", False),
        rbf_learnable_sigma=cfg.get("rbf_learnable_sigma", False),
        rbf_sigma_factor=cfg.get("rbf_sigma_factor", 1.0),
        rbf_short_min=cfg.get("rbf_short_min", 0.0),
        rbf_short_max=cfg.get("rbf_short_max", 8.0),
        rbf_short_kernels=cfg.get("rbf_short_kernels", 6),
        rbf_medium_min=cfg.get("rbf_medium_min", 5.0),
        rbf_medium_max=cfg.get("rbf_medium_max", 15.0),
        rbf_medium_kernels=cfg.get("rbf_medium_kernels", 8),
        rbf_long_min=cfg.get("rbf_long_min", 10.0),
        rbf_long_max=cfg.get("rbf_long_max", 25.0),
        rbf_long_kernels=cfg.get("rbf_long_kernels", 6),
        rbf_scale_init=cfg.get("rbf_scale_init", 0.1),

        # Neighbor selection
        selection_method=cfg.get("selection_method", "knn"),
        threshold_target_sparsity=cfg.get("threshold_target_sparsity", 0.2),
        threshold_sparsity_weight=cfg.get("threshold_sparsity_weight", 1e-4),
        cfg=cfg,
        distance_eps=cfg.get("distance_eps", 1e-6),
    ).to(device)
    return model
