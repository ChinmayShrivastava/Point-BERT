import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
from .dvae import Group
from .dvae import DiscreteVAE, Encoder

from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from enum import Enum, auto
import math



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        qkv_bias=False, 
        qk_scale=None, 
        attn_drop=0., 
        proj_drop=0., 
        use_lora=False, 
        lora_rank=4, 
        lora_alpha=32
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_lora = use_lora

        # Original layers
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Add LoRA layers
        if use_lora:
            self.lora_qkv = LoRALayer(dim, dim * 3, rank=lora_rank, alpha=lora_alpha)
            self.lora_proj = LoRALayer(dim, dim, rank=lora_rank, alpha=lora_alpha)

    def forward(self, x):
        B, N, C = x.shape
        
        # Combine original and LoRA outputs for qkv
        qkv = self.qkv(x)
        if self.use_lora:
            qkv = qkv + self.lora_qkv(x)
        
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Combine original and LoRA outputs for projection
        x = self.proj(x)
        if self.use_lora:
            x = x + self.lora_proj(x)
            
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_lora=False, lora_rank=4, lora_alpha=32):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            use_lora=use_lora, lora_rank=lora_rank, lora_alpha=lora_alpha
        )
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., use_lora=False, lora_rank=4, lora_alpha=32):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x

class LossType(Enum):
    MSE = 'mse'
    L1 = 'l1'
    SmoothL1 = 'smooth_l1'
    HUBER = 'huber'
    WEIGHTED_PRESSURE = 'weighted_pressure'

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros(in_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling

@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, dropout: float = 0.1, loss_type: LossType = LossType.MSE,
                 use_lora: bool = False, lora_rank: int = 4, lora_alpha: float = 32,
                 residuals: bool = False, **kwargs):
        super().__init__()
        self.config = config
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.npoints = config.npoints
        self.dropout = dropout
        self.loss_type = loss_type

        self.trans_dim = config.trans_dim
        self.depth = config.depth 
        self.drop_path_rate = config.drop_path_rate 
        self.cls_dim = config.cls_dim 
        self.num_heads = config.num_heads 

        self.group_size = config.group_size
        self.num_group = config.num_group
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
            use_lora = use_lora,
            lora_rank = lora_rank,
            lora_alpha = lora_alpha
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        
        self.residuals = residuals
        if self.residuals:
            self.finetuning_head = nn.Sequential(
                nn.Linear(self.trans_dim * 3, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(256, 1),  # Output 1 pressure value per point
                nn.Sigmoid()  # Output range [0, 1]
            )
        else:
            self.finetuning_head = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(256, 1),  # Output 1 pressure value per point
                nn.Sigmoid()  # Output range [0, 1]
            )

        self.build_loss_func()
        
        # Add dtype conversion in forward method
        self.register_buffer('dtype_holder', torch.zeros(1))  # Used to match dtype
        
        # Add device tracking
        self.register_buffer('device_holder', torch.zeros(1))  # Used to track device
        
        self.loss_weight_variance = 0.1

    def prepare_for_lora_training(self):
        fn_prepare_for_lora_training(self)
        
    def build_loss_func(self):
        if self.loss_type == LossType.MSE:
            self.loss_ce = nn.MSELoss()
        elif self.loss_type == LossType.L1:
            self.loss_ce = nn.L1Loss()
        elif self.loss_type == LossType.SmoothL1:
            self.loss_ce = nn.SmoothL1Loss()
        elif self.loss_type == LossType.HUBER:
            self.loss_ce = nn.HuberLoss()
        elif self.loss_type == LossType.WEIGHTED_PRESSURE:
            # Initialize with MSE loss but will apply weights during calculation
            self.loss_ce = nn.MSELoss(reduction='none')
            # Default number of bins for pressure histogram
            self.pressure_bins = 20
            # Store histogram of pressure values for weight calculation
            self.register_buffer('pressure_histogram', torch.zeros(self.pressure_bins))
            self.register_buffer('total_samples', torch.zeros(1))
            # Smoothing factor to avoid division by zero
            self.histogram_smoothing = 0.1
                
    
    def get_loss(self, pred, gt):
        # Ensure both tensors are on the same device
        gt = gt.to(device=pred.device, dtype=pred.dtype)
        
        if self.loss_type == LossType.WEIGHTED_PRESSURE:
            # Update histogram of pressure values (during training)
            if self.training:
                # Create histogram of current batch
                hist = torch.histc(gt, bins=self.pressure_bins, min=0.0, max=1.0)
                # Update running histogram with exponential moving average
                alpha = 0.9  # Smoothing factor for histogram update
                self.pressure_histogram = alpha * self.pressure_histogram + (1 - alpha) * hist
                self.total_samples += gt.numel()
            
            # Calculate weights based on inverse frequency
            # Get bin indices for each gt value
            bin_indices = (gt * self.pressure_bins).long().clamp(0, self.pressure_bins - 1)
            
            # Get weights from histogram (inverse frequency)
            if self.total_samples > 0:
                # Normalize histogram to get probabilities
                probs = (self.pressure_histogram + self.histogram_smoothing) / (self.total_samples + self.pressure_bins * self.histogram_smoothing)
                # Inverse frequency weighting
                weights = 1.0 / (probs[bin_indices] * self.pressure_bins)
                # Normalize weights to have mean=1
                weights = weights / weights.mean()
            else:
                # If no samples seen yet, use uniform weights
                weights = torch.ones_like(gt)
            
            # Calculate weighted MSE loss
            element_wise_loss = self.loss_ce(pred, gt)
            weighted_loss = (element_wise_loss * weights).mean()
            
            # Add variance matching term if configured
            if hasattr(self, 'loss_weight_variance') and self.loss_weight_variance > 0:
                gt_mean = gt.mean()
                gt_var = ((gt - gt_mean) ** 2).mean()
                pred_mean = pred.mean()
                pred_var = ((pred - pred_mean) ** 2).mean()
                
                variance_loss = torch.abs(pred_var - gt_var)
                return weighted_loss + self.loss_weight_variance * variance_loss
            else:
                return weighted_loss
        elif self.loss_type == LossType.HUBER:
            base_loss = F.huber_loss(pred, gt)
            if hasattr(self, 'loss_weight_variance') and self.loss_weight_variance > 0:
                gt_mean = gt.mean()
                gt_var = ((gt - gt_mean) ** 2).mean()
                pred_mean = pred.mean()
                pred_var = ((pred - pred_mean) ** 2).mean()
                
                variance_loss = torch.abs(pred_var - gt_var)
                return base_loss + self.loss_weight_variance * variance_loss
            else:
                return base_loss
        else:
            return self.loss_ce(pred, gt)

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        
        # Handle both old and new checkpoint formats
        if 'model_state_dict' in ckpt:
            # New format from finetune_few_shot.py
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['model_state_dict'].items()}
        else:
            # Original format
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            for k in list(base_ckpt.keys()):
                if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                    base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger = 'Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger = 'Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger = 'Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger = 'Transformer'
            )

        print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger = 'Transformer')

    def forward(self, input):
        # Convert input to correct dtype and device
        input = input.to(device=self.device_holder.device, dtype=self.dtype_holder.dtype)
        
        B, N, _ = input.shape
        
        # Group and encode the input
        neighborhood, center = self.group_divider(input) # B N G 3, B N 3
        group_input_tokens = self.encoder(neighborhood)  # B G C
        group_input_tokens = self.reduce_dim(group_input_tokens)
        
        # Add cls token and position embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)
        pos = self.pos_embed(center)
        
        # Concatenate tokens and process through transformer
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = self.blocks(x, pos)
        
        x = self.norm(x)
        
        # Get global features
        global_features = x[:, 0]  # Take cls token features (B, C)
        group_features = x[:, 1:]  # Take group token features (B, G, C)
        
        # Interpolate group features back to original points
        group_points = center  # (B, G, 3)
        input_points = input  # (B, N, 3)
        
        # Calculate distances between input points and group centers
        input_points_expanded = input_points.unsqueeze(2)  # (B, N, 1, 3)
        group_points_expanded = group_points.unsqueeze(1)  # (B, 1, G, 3)
        distances = torch.sum((input_points_expanded - group_points_expanded) ** 2, dim=-1)  # (B, N, G)
        
        # Get k nearest group centers for each input point
        # k = 3  # number of nearest neighbors
        # Since we only need distances for the weights, we don't need to calculate indices
        weights = F.softmax(-distances, dim=-1)  # (B, N, G)
        weights = weights.unsqueeze(-1)  # (B, N, G, 1)
        
        # Expand group features for each input point
        group_features_expanded = group_features.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, G, C)
        
        # Weighted sum of features
        interpolated_features = (group_features_expanded * weights).sum(dim=2)  # (B, N, C)
        
        # Expand global features and concatenate
        global_features_expanded = global_features.unsqueeze(1).expand(-1, N, -1)  # (B, N, C)
        
        if self.residuals:
            # use self.pos_embed to embed all the input points
            input_pos_embed = self.pos_embed(input)  # (B, N, C)
            
            combined_features = torch.cat([global_features_expanded, interpolated_features, input_pos_embed], dim=-1)  # (B, N, 3C)
        else:
            combined_features = torch.cat([global_features_expanded, interpolated_features], dim=-1)  # (B, N, 2C)
        
        # Generate displacements
        finetuned_output = self.finetuning_head(combined_features)
        
        return finetuned_output

class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.trans_dim = config.transformer_config.trans_dim 
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate 
        self.cls_dim = config.transformer_config.cls_dim 
        self.replace_pob = config.transformer_config.replace_pob 
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[Transformer args] {config.transformer_config}', logger = 'dVAE BERT') 
        # define the encoder
        self.encoder_dims =  config.dvae_config.encoder_dims 
        self.encoder = Encoder(encoder_channel = self.encoder_dims) 
        # bridge encoder and transformer 
        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim) 
        try:
            self.mask_rand = config.mask_rand
        except:
            self.mask_rand = False
        
        # define the learnable tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        # pos embedding for each patch 
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        # define the transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )
        # layer norm
        self.norm = nn.LayerNorm(self.trans_dim)
        # head for token classification
        self.num_tokens = config.dvae_config.num_tokens
        self.lm_head = nn.Linear(self.trans_dim, self.num_tokens)
        # head for cls contrast
        self.cls_head = nn.Sequential(
            nn.Linear(self.trans_dim, self.cls_dim),
            nn.GELU(),
            nn.Linear(self.cls_dim, self.cls_dim)
        )  
        # initialize the learnable tokens
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _prepare_encoder(self, dvae_ckpt):
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        encoder_ckpt = {k.replace("encoder.", ""): v for k, v in base_ckpt.items() if 'encoder' in k}

        self.encoder.load_state_dict(encoder_ckpt, strict=True)
        print_log(f'[Encoder] Successful Loading the ckpt for encoder from {dvae_ckpt}', logger = 'dVAE BERT')


    def _mask_center(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0) # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p =2 ,dim = -1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0] # G
            ratio = random.uniform(self.mask_ratio[0], self.mask_ratio[1])
            mask_num = int(ratio * len(idx))

            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())
        
        bool_masked_pos = torch.stack(mask_idx).to(center.device) # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()
        
        ratio = random.random() * (self.mask_ratio[1] - self.mask_ratio[0]) + self.mask_ratio[0]
        bool_masked_pos = (torch.rand(center.shape[:2]) < ratio).bool().to(center.device)

        return bool_masked_pos


    def _random_replace(self, group_input_tokens, bool_masked_pos, noaug = False):
        '''
            group_input_tokens : B G C
            bool_masked_pos : B G  
            -----------------
            replaced_group_input_tokens: B G C
        '''
        # skip replace
        if noaug or self.replace_pob == 0:
            return group_input_tokens, bool_masked_pos
        
        replace_mask = (torch.rand(group_input_tokens.shape[:2]) < self.replace_pob).to(bool_masked_pos.device).bool()
        replace_mask = (replace_mask & ~bool_masked_pos)  # do not replace the mask pos

        overall_mask = (replace_mask + bool_masked_pos).bool().to(bool_masked_pos.device)  #  True for flake input

        detached_group_input_tokens = group_input_tokens.detach()
        flatten_group_input_tokens = detached_group_input_tokens.reshape(detached_group_input_tokens.size(0) * detached_group_input_tokens.size(1), detached_group_input_tokens.size(2))
        idx = torch.randperm(flatten_group_input_tokens.shape[0])
        shuffled_group_input_tokens = flatten_group_input_tokens[idx].reshape(detached_group_input_tokens.size(0), detached_group_input_tokens.size(1), detached_group_input_tokens.size(2))

        replace_mask = replace_mask.unsqueeze(-1).type_as(detached_group_input_tokens)
        replaced_group_input_tokens = group_input_tokens * (1 - replace_mask) + shuffled_group_input_tokens * replace_mask
        return replaced_group_input_tokens, overall_mask

    def forward(self, neighborhood, center, return_all_tokens = False, only_cls_tokens = False, noaug = False):
        # generate mask
        if self.mask_rand:
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        else:
            bool_masked_pos = self._mask_center(center, noaug = noaug) # B G
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # replace the point
        replaced_group_input_tokens, overall_mask = self._random_replace(group_input_tokens, bool_masked_pos.clone(), noaug = noaug)
        batch_size, seq_len, _ = replaced_group_input_tokens.size()
        # prepare cls and mask
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  
        cls_pos = self.cls_pos.expand(batch_size, -1, -1)  
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        # mask the input tokens
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        maksed_group_input_tokens = replaced_group_input_tokens * (1 - w) + mask_token * w
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, maksed_group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        # only return the cls feature, for moco contrast
        if only_cls_tokens:
            return self.cls_head(x[:, 0])
        logits = self.lm_head(x[:, 1:])
        if return_all_tokens:
            return self.cls_head(x[:, 0]), logits
        else:
            # return the flake logits
            return self.cls_head(x[:, 0]), logits[~overall_mask], logits[overall_mask], overall_mask # reduce the Batch dim        

@MODELS.register_module()
class Point_BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_BERT] build dVAE_BERT ...', logger ='Point_BERT')
        self.config = config
        self.m = config.m
        self.T = config.T
        self.K = config.K

        self.moco_loss = config.transformer_config.moco_loss
        self.dvae_loss = config.transformer_config.dvae_loss
        self.cutmix_loss = config.transformer_config.cutmix_loss

        self.return_all_tokens = config.transformer_config.return_all_tokens
        if self.return_all_tokens:
            print_log(f'[Point_BERT] Point_BERT calc the loss for all token ...', logger ='Point_BERT')
        else:
            print_log(f'[Point_BERT] Point_BERT [NOT] calc the loss for all token ...', logger ='Point_BERT')
        
        self.transformer_q = MaskTransformer(config)
        self.transformer_q._prepare_encoder(self.config.dvae_config.ckpt)
        
        self.transformer_k = MaskTransformer(config)
        for param_q, param_k in zip(self.transformer_q.parameters(), self.transformer_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.dvae = DiscreteVAE(config.dvae_config)
        self._prepare_dvae()

        for param in self.dvae.parameters():
            param.requires_grad = False

        self.group_size = config.dvae_config.group_size
        self.num_group = config.dvae_config.num_group

        print_log(f'[Point_BERT Group] cutmix_BERT divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_BERT')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # create the queue
        self.register_buffer("queue", torch.randn(self.transformer_q.cls_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # loss
        self.build_loss_func()

    def _prepare_dvae(self):
        dvae_ckpt = self.config.dvae_config.ckpt
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        self.dvae.load_state_dict(base_ckpt, strict=True)
        print_log(f'[dVAE] Successful Loading the ckpt for dvae from {dvae_ckpt}', logger ='Point_BERT')


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.transformer_q.parameters(), self.transformer_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_batch = nn.CrossEntropyLoss(reduction='none')

    def forward_eval(self, pts):
        with torch.no_grad():
            neighborhood, center = self.group_divider(pts)
            cls_feature = self.transformer_q(neighborhood, center, only_cls_tokens = True, noaug = True)
            return cls_feature

    def _mixup_pc(self, neighborhood, center, dvae_label):
        '''
            neighborhood : B G M 3
            center: B G 3
            dvae_label: B G
            ----------------------
            mixup_ratio: /alpha:
                mixup_label = alpha * origin + (1 - alpha) * flip

        '''
        mixup_ratio = torch.rand(neighborhood.size(0))
        mixup_mask = torch.rand(neighborhood.shape[:2]) < mixup_ratio.unsqueeze(-1)
        mixup_mask = mixup_mask.type_as(neighborhood)
        mixup_neighborhood = neighborhood * mixup_mask.unsqueeze(-1).unsqueeze(-1) + neighborhood.flip(0) * (1 - mixup_mask.unsqueeze(-1).unsqueeze(-1))
        mixup_center = center * mixup_mask.unsqueeze(-1) + center.flip(0) * (1 - mixup_mask.unsqueeze(-1))
        mixup_dvae_label = dvae_label * mixup_mask + dvae_label.flip(0) * (1 - mixup_mask)

        return mixup_ratio.to(neighborhood.device), mixup_neighborhood, mixup_center, mixup_dvae_label.long()


    def forward(self, pts, noaug = False, **kwargs):
        if noaug:
            return self.forward_eval(pts)
        else:
            # divide the point cloud in the same form. This is important
            neighborhood, center = self.group_divider(pts)
            # produce the gt point tokens
            with torch.no_grad():
                gt_logits = self.dvae.encoder(neighborhood) 
                gt_logits = self.dvae.dgcnn_1(gt_logits, center) #  B G N
                dvae_label = gt_logits.argmax(-1).long() # B G 
            # forward the query model in mask style 1.
            if self.return_all_tokens:
                q_cls_feature, logits = self.transformer_q(neighborhood, center, return_all_tokens = self.return_all_tokens) # logits :  N G C 
            else:
                q_cls_feature, real_logits, flake_logits, mask = self.transformer_q(neighborhood, center, return_all_tokens = self.return_all_tokens) # logits :  N' C where N' is the mask.sum() 
            q_cls_feature = nn.functional.normalize(q_cls_feature, dim=1)

            mixup_ratio, mixup_neighborhood, mixup_center, mix_dvae_label = self._mixup_pc(neighborhood, center, dvae_label)
            if self.return_all_tokens:
                mixup_cls_feature, mixup_logits = self.transformer_q(mixup_neighborhood, mixup_center, return_all_tokens = self.return_all_tokens) 
            else:
                mixup_cls_feature, mixup_real_logits, mixup_flake_logits, mixup_mask = self.transformer_q(mixup_neighborhood, mixup_center, return_all_tokens = self.return_all_tokens)
            mixup_cls_feature = nn.functional.normalize(mixup_cls_feature, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder
                k_cls_feature = self.transformer_k(neighborhood, center, only_cls_tokens = True)  # keys: NxC
                k_cls_feature = nn.functional.normalize(k_cls_feature, dim=1)

            if self.moco_loss:
                # ce loss with moco contrast
                l_pos = torch.einsum('nc, nc->n', [q_cls_feature, k_cls_feature]).unsqueeze(-1) # n 1 
                l_neg = torch.einsum('nc, ck->nk', [q_cls_feature, self.queue.clone().detach()]) # n k
                ce_logits = torch.cat([l_pos, l_neg], dim=1)
                ce_logits /= self.T
                labels = torch.zeros(l_pos.shape[0], dtype=torch.long).to(pts.device)
                moco_loss = self.loss_ce(ce_logits, labels)
            else:
                moco_loss = torch.tensor(0.).to(pts.device)

            if self.dvae_loss:
                if self.return_all_tokens:
                    dvae_loss = self.loss_ce(logits.reshape(-1, logits.size(-1)), dvae_label.reshape(-1,)) + \
                             self.loss_ce(mixup_logits.reshape(-1, mixup_logits.size(-1)), mix_dvae_label.reshape(-1,))
                else:
                    dvae_loss = self.loss_ce(flake_logits, dvae_label[mask]) + \
                        self.loss_ce(mixup_flake_logits, mix_dvae_label[mixup_mask])
            else:
                dvae_loss = torch.tensor(0.).to(pts.device)

            if self.cutmix_loss:                
                l_pos = torch.einsum('nc, mc->nm', [mixup_cls_feature, k_cls_feature]) # n n
                l_neg = torch.einsum('nc, ck->nk', [mixup_cls_feature, self.queue.clone().detach()]) # n k
                ce_logits = torch.cat([l_pos, l_neg], dim=1)
                ce_logits /= self.T
                labels = torch.arange(l_pos.shape[0], dtype=torch.long).to(pts.device)
                cutmix_loss = (mixup_ratio * self.loss_ce_batch(ce_logits, labels) + (1 - mixup_ratio) * self.loss_ce_batch(ce_logits, labels.flip(0))).mean()
            else:
                cutmix_loss = torch.tensor(0.).to(pts.device)
            self._dequeue_and_enqueue(k_cls_feature)
            return moco_loss + dvae_loss, cutmix_loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def fn_prepare_for_lora_training(model):
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze LoRA parameters
    for module in model.modules():
        if isinstance(module, LoRALayer):
            for param in module.parameters():
                param.requires_grad = True
                
    # Unfreeze displacement head parameters since it's new
    for param in model.finetuning_head.parameters():
        param.requires_grad = True