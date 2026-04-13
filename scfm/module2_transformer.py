"""
=============================================================================
scFM Tutorial — Module 2: The Masked Attention Transformer
=============================================================================

In Module 1, we built the input pipeline: vocabulary, value binning, and
the three-part embedding (gene + expression + condition). Now we implement
the CORE of scFM: the transformer with its specialized attention masking
for non-sequential gene expression data.

Key learning objectives in this module:
  1. Why standard transformer attention must be MODIFIED for single-cell data
  2. The scFM attention mask and how it enables generative pretraining
  3. Building the full transformer block (multi-head attention + FFN)
  4. How the <cls> token creates a cell-level representation
  5. The pretraining objective: predicting masked gene expression values

Architecture recap from paper (Methods → Implementation details):
  - Embedding size D = 512
  - 12 stacked transformer blocks
  - 8 attention heads per block
  - Hidden layer size in FFN = 512 (same as D; typical choice is 2x-4x but
    authors use 1x here — keeps parameter count manageable)
  - FlashAttention for GPU efficiency at long sequence lengths

=============================================================================
SECTION 1: The Attention Masking Problem
=============================================================================

WHY STANDARD CAUSAL (GPT) MASKING DOESN'T WORK:
  In language GPT, tokens have a natural order: token 1 → token 2 → token 3
  Causal masking prevents token j from attending to token j+1 (future).
  This is meaningful because word ORDER conveys meaning.

  In scRNA-seq:
    Cell = {GAPDH: 45, TP53: 12, MYC: 89, ...}   ← a SET, not a sequence
    Shuffling genes doesn't change the cell's biological meaning.
    There is NO "next gene" to predict in the causal sense.

WHY STANDARD BERT MASKING DOESN'T FULLY WORK EITHER:
  BERT randomly masks tokens and predicts them from bidirectional context.
  Problem: BERT is an encoder only — it can't GENERATE new sequences.
  scFM needs to GENERATE expression profiles (cell-prompt generation).

THE scFM SOLUTION: A unified generative mask that:
  - Groups tokens into "known" (prompt) and "unknown" (to generate)
  - Unknown tokens can only attend to known tokens + themselves
  - Known tokens attend to all other known tokens (bidirectional within known)
  - This allows iterative generation: predict unknown genes in batches,
    then fold them into "known" for the next round

This is formalized in Eq. 11 of the paper:
  a[i,j] = 0       if j is NOT an unknown gene  (attend freely to known genes)
  a[i,j] = 0       if i == j AND j is unknown    (gene attends to itself)
  a[i,j] = -inf    if i != j AND j is unknown    (block unknown-to-unknown)

Adding -inf to the raw attention logit → 0 after softmax → no attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple


# =============================================================================
# SECTION 2: Building the Attention Mask
# =============================================================================

def build_scgpt_attention_mask(
    seq_len: int,
    known_mask: torch.BoolTensor,  # True = known gene, False = unknown
    device: torch.device
) -> torch.Tensor:
    """
    Build the scFM specialized attention mask.
    
    Args:
        seq_len:    Total sequence length M (including <cls> token)
        known_mask: Bool tensor of shape [M], True = known/prompt gene
        device:     Target device
    
    Returns:
        attn_mask: Float tensor [M, M], 0 = attend, -inf = block
    
    PERFORMANCE NOTE on mask construction:
      We use float('-inf') rather than a large negative (like -1e9) because:
      - float('-inf') + any_finite = float('-inf'), which is exact
      - After softmax, exp(-inf) = 0 exactly (no floating point residue)
      - Using -1e9 can cause numerical instabilities with float16/bf16
      Always prefer true -inf for masking in production code.
    
    SHAPE NOTE:
      PyTorch's MultiheadAttention expects attn_mask of shape:
      - [tgt_len, src_len] (applied to all batches/heads equally), OR
      - [batch*n_heads, tgt_len, src_len] (per-head mask)
      We use the simpler [M, M] form here (broadcast across batch/heads).
    """
    # Start with all zeros (all attention allowed)
    mask = torch.zeros(seq_len, seq_len, device=device)
    
    # Find positions of unknown genes
    # known_mask: True = known, so unknown positions are ~known_mask
    unknown_positions = (~known_mask).nonzero(as_tuple=True)[0]
    
    # For each pair (i, j) where j is unknown AND i != j: block attention
    for j in unknown_positions:
        # Block the entire column j (no gene can see this unknown gene)
        mask[:, j] = float('-inf')
        # But allow the gene to attend to itself (diagonal = 0)
        mask[j, j] = 0.0
    
    return mask


def build_scgpt_attention_mask_vectorized(
    seq_len: int,
    known_mask: torch.BoolTensor,
    device: torch.device
) -> torch.Tensor:
    """
    Vectorized (faster) version of the attention mask builder.
    
    PERFORMANCE TIP: Avoid Python loops when building masks.
    For seq_len=1200, the loop version runs O(n_unknown) iterations.
    This vectorized version is O(1) in Python (all ops are CUDA kernels).
    
    This matters during training when you're building a new mask every
    forward pass. At batch_size=32, seq_len=1200, you want mask creation
    to be sub-millisecond.
    """
    # Create unknown column mask: shape [M] where True = unknown position
    unknown_col = ~known_mask  # [M]
    
    # Broadcast to [M, M]: column j is blocked for all rows
    # unknown_col[None, :] has shape [1, M] → broadcasts to [M, M]
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask[:, unknown_col] = float('-inf')
    
    # Restore diagonal for unknown positions (self-attention allowed)
    diag_indices = torch.arange(seq_len, device=device)
    mask[diag_indices, diag_indices] = 0.0
    
    return mask


def demonstrate_attention_mask():
    """
    Visualize the scFM attention mask with a small example.
    
    Tokens: [<cls>, Gene1(known), Gene2(known), Gene3(unknown), Gene4(unknown)]
    Expected pattern:
      - <cls> and Gene1/2 (known): can attend to all known tokens
      - Gene3, Gene4 (unknown): blocked from attending to each other
        but can attend to known tokens and themselves
    """
    print("=" * 60)
    print("scFM Attention Mask Demonstration")
    print("=" * 60)
    
    seq_len = 5
    # Token order: [<cls>, gene1(known), gene2(known), gene3(unknown), gene4(unknown)]
    known_mask = torch.tensor([True, True, True, False, False])
    token_names = ["<cls>", "G1(K)", "G2(K)", "G3(U)", "G4(U)"]
    
    mask = build_scgpt_attention_mask_vectorized(seq_len, known_mask, device="cpu")
    
    print("\nAttention mask (0=attend, -inf=blocked):")
    print("Rows=queries, Cols=keys")
    print(f"{'':10}", end="")
    for name in token_names:
        print(f"{name:10}", end="")
    print()
    
    for i, row_name in enumerate(token_names):
        print(f"{row_name:10}", end="")
        for j in range(seq_len):
            val = mask[i, j].item()
            symbol = "  attend " if val == 0.0 else "  BLOCK  "
            print(f"{symbol:10}", end="")
        print()
    
    print("\nKey observations:")
    print("  1. G3(U) and G4(U) are blocked from each other (unknown→unknown)")
    print("  2. G3(U) and G4(U) CAN attend to known tokens (<cls>, G1, G2)")
    print("  3. G3(U) and G4(U) CAN attend to themselves (diagonal)")
    print("  4. Known tokens attend freely to all other known tokens")
    print()
    print("This means: when predicting G3's expression, the model uses")
    print("G1 and G2 expression as context, but does NOT cheat by looking")
    print("at G4 (which is also being predicted in this round).")


# =============================================================================
# SECTION 3: Scaled Dot-Product Attention (from scratch)
# =============================================================================

class ScaledDotProductAttention(nn.Module):
    """
    The fundamental attention operation from "Attention is All You Need".
    
    Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) + mask ) * V
    
    WHY SCALE BY sqrt(d_k)?
      Without scaling, for large d_k, QK^T values grow proportional to d_k.
      Large values push softmax into saturation (near 0 or 1), causing
      vanishing gradients. Scaling by 1/sqrt(d_k) keeps variance ≈ 1.
      
      Example: d_k=64 → scale=0.125, d_k=512 → scale=0.044
      At larger d_k, scaling is MORE important to prevent saturation.
    
    DROPOUT IN ATTENTION:
      Dropout is applied to the attention WEIGHTS (after softmax).
      This randomly zeros out entire gene-gene attention connections
      during training, acting as a regularizer for gene interaction learning.
      
      Typical value: dropout=0.1 (10% of attention connections dropped)
      Higher dropout → stronger regularization but slower convergence.
    """
    
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        Q: torch.Tensor,           # [batch, n_heads, seq_len, d_k]
        K: torch.Tensor,           # [batch, n_heads, seq_len, d_k]
        V: torch.Tensor,           # [batch, n_heads, seq_len, d_v]
        attn_mask: Optional[torch.Tensor] = None,  # [seq_len, seq_len]
        key_padding_mask: Optional[torch.Tensor] = None  # [batch, seq_len]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: [batch, n_heads, seq_len, d_v]
            attn_weights: [batch, n_heads, seq_len, seq_len] (for GRN analysis!)
        
        NOTE ON SAVING ATTENTION WEIGHTS:
          The paper's GRN inference (Fig. 6) uses raw attention scores from
          the last transformer layer. We return these weights here so callers
          can save them for downstream analysis.
          
          During pretraining, you won't need them. But at inference time for
          GRN discovery, you want to set model.eval() and capture these weights.
        """
        d_k = Q.size(-1)
        
        # Compute raw attention scores: [batch, n_heads, seq_len, seq_len]
        # bmm (batch matrix multiply) handles the batched QK^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply structural attention mask (scFM's known/unknown mask)
        # attn_mask shape: [seq_len, seq_len] → broadcast to [batch, n_heads, seq_len, seq_len]
        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply padding mask (masks out <pad> tokens)
        # key_padding_mask: [batch, seq_len], True = should be masked
        if key_padding_mask is not None:
            # Expand: [batch, 1, 1, seq_len] to broadcast over queries and heads
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax over the key dimension → attention weights sum to 1
        attn_weights = F.softmax(scores, dim=-1)
        
        # Handle the case where a row is all -inf (e.g., a padding token)
        # After softmax, all-inf rows become NaN; replace with zeros
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # Apply dropout to attention weights (regularization)
        attn_weights_dropped = self.dropout(attn_weights)
        
        # Weighted sum of values: [batch, n_heads, seq_len, d_v]
        output = torch.matmul(attn_weights_dropped, V)
        
        return output, attn_weights  # Return undropped weights for analysis


# =============================================================================
# SECTION 4: Multi-Head Attention
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention allows the model to attend to different aspects
    of gene relationships simultaneously with different "views".
    
    With n_heads=8 and d_model=512:
      - Each head has d_k = d_v = d_model / n_heads = 64 dimensions
      - Head 1 might learn: "which genes are co-expressed with this gene"
      - Head 2 might learn: "which genes are in the same pathway"
      - Head 3 might learn: "which genes are regulated by this gene"
      - ... and so on
    
    WHY 8 HEADS?
      This is an empirical sweet spot. Too few heads: limited expressivity.
      Too many heads: each head has tiny d_k → each head captures less info.
      For d_model=512, 8 heads gives d_k=64, which is comfortable.
      Common choices: 4, 8, 12, 16 heads.
    
    PARAMETER COUNT:
      4 projection matrices (Q, K, V, O), each [d_model, d_model]
      Total: 4 × 512 × 512 = 1,048,576 ≈ 1M parameters per MHA layer
      For 12 layers: 12M parameters just in MHA projections
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 64 for d_model=512, n_heads=8
        
        # Linear projections for Q, K, V, and output
        # These are "learned linear transformations" — they rotate/scale the
        # input embedding space to a space useful for attention
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)  # Output projection
        
        # WHY bias=False for Q,K,V? The bias in the Q and K projections would
        # add a constant to all attention scores, shifting the distribution.
        # In practice, omitting it has little effect but slightly reduces params.
        # W_o keeps bias as it provides flexibility in the output projection.
        
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,       # [batch, seq_len, d_model]
        key: torch.Tensor,         # [batch, seq_len, d_model]
        value: torch.Tensor,       # [batch, seq_len, d_model]
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, seq_len, _ = query.shape
        
        # Project inputs to Q, K, V
        # Shape: [batch, seq_len, d_model]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into multiple heads and reshape:
        # [batch, seq_len, d_model] → [batch, n_heads, seq_len, d_k]
        # This is a RESHAPE, not a new computation — same parameters
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention for all heads in parallel
        attn_output, attn_weights = self.attention(Q, K, V, attn_mask, key_padding_mask)
        # attn_output: [batch, n_heads, seq_len, d_k]
        # attn_weights: [batch, n_heads, seq_len, seq_len]
        
        # Concatenate heads: [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Final output projection
        output = self.W_o(attn_output)
        
        if return_attn_weights:
            return output, attn_weights  # Return for GRN analysis
        return output, None


# =============================================================================
# SECTION 5: Feed-Forward Network
# =============================================================================

class PositionwiseFeedForward(nn.Module):
    """
    The FFN in each transformer block: a 2-layer MLP applied to each position.
    
    FFN(x) = W2 * GELU( W1 * x + b1 ) + b2
    
    The FFN is applied independently to each gene token, allowing the model
    to transform each gene's representation after attention has "mixed" information
    from other genes.
    
    HIDDEN DIMENSION CHOICE:
      Standard transformer: d_ff = 4 × d_model (e.g., 4×512=2048)
      scFM paper (Methods → Implementation): d_ff = 512 = d_model (1×)
      
      Why did the authors use 1× instead of 4×?
      - Memory efficiency: with seq_len=1200 and large batches, FFN is a
        memory bottleneck. 1× keeps it feasible on available GPUs.
      - The authors may have found diminishing returns from larger FFN
        given the nature of gene expression data.
      
      PERFORMANCE TIP: If you have more GPU memory, try d_ff = 2048 (4×d_model).
      This is a common hyperparameter to tune for performance improvement.
    
    GELU vs ReLU:
      GELU (Gaussian Error Linear Unit) is smoother than ReLU and empirically
      performs better in transformer models. It approximates:
      GELU(x) ≈ x * Φ(x) where Φ is the Gaussian CDF.
      Near x=0, it allows small negative activations, which can improve
      gradient flow. Modern models (BERT, GPT, etc.) all use GELU.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 512,     # Paper uses 512 = 1×d_model (unusual choice)
        dropout: float = 0.1
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        x = self.linear1(x)      # [batch, seq_len, d_ff]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)      # [batch, seq_len, d_model]
        return x


# =============================================================================
# SECTION 6: Full Transformer Block with Pre-LayerNorm
# =============================================================================

class scFMTransformerBlock(nn.Module):
    """
    One transformer block = MultiHeadAttention + FFN + residual connections.
    
    ARCHITECTURE CHOICE: Pre-LayerNorm vs Post-LayerNorm
    
    Standard (Post-LN, original Transformer paper):
      x = LayerNorm(x + Sublayer(x))
    
    Pre-LN (used in many modern transformers including GPT-2, GPT-3):
      x = x + Sublayer(LayerNorm(x))
    
    WHY PRE-LN IS BETTER for deep models:
      - With Post-LN, gradients can explode in very deep networks because
        the residual path skips LayerNorm.
      - Pre-LN keeps the residual path "clean" (unnormalized), which makes
        training more stable without learning rate warmup.
      - For 12 layers (scFM), Pre-LN is the safer choice.
      
    We implement Pre-LN here. The paper doesn't explicitly specify which
    variant was used, but modern implementations of scFM use Pre-LN.
    
    RESIDUAL CONNECTIONS:
      x = x + Sublayer(...)
      
      Why? They allow gradients to flow directly back through the network
      without passing through the sublayer. This prevents vanishing gradients
      in deep networks (the core insight of ResNets, applied to transformers).
      Each layer learns a DELTA to add to the current representation,
      not a full new representation.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # LayerNorm: normalizes across the d_model dimension for each token
        # Pre-LN: normalize BEFORE attention and FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout on sublayer outputs (before residual addition)
        # This is different from attention dropout and FFN internal dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,           # [batch, seq_len, d_model]
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # Pre-LN self-attention with residual connection
        normed = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            normed, normed, normed,  # Self-attention: Q=K=V=same input
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            return_attn_weights=return_attn_weights
        )
        x = x + self.dropout(attn_out)  # Residual addition
        
        # Pre-LN FFN with residual connection
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)   # Residual addition
        
        return x, attn_weights


# =============================================================================
# SECTION 7: The Full scFM Transformer Model
# =============================================================================

class scFMTransformer(nn.Module):
    """
    The full scFM transformer: stacked transformer blocks.
    
    This is the CORE of scFM. It takes the three-part embeddings from
    Module 1 and produces rich gene-level and cell-level representations.
    
    Architecture:
      Input: h^(0) = embedding_output  [batch, seq_len, d_model]
      For each layer l:
        h^(l) = transformer_block(h^(l-1))
      Output: h^(n) [batch, seq_len, d_model]
    
    Cell representation:
      h_cell = h^(n)[0]  ← the <cls> token at position 0
    Gene representations:
      h_genes = h^(n)[1:]  ← all other positions
    
    PARAMETER COUNT for 12 layers:
      Per block:
        MHA: 4 × 512² = 1,048,576
        FFN: 2 × 512² = 524,288
        LayerNorms: 2 × 2 × 512 = 2,048
      Total per block: ~1.57M
      Total for 12 blocks: ~18.9M
      Plus embeddings: ~10.5M (vocab×512)
      Grand total: ~29-30M parameters (matches paper's scale)
    
    DEPTH CHOICE (12 layers):
      Fewer layers (6): faster training, less representational capacity
      More layers (24): GPT-2/BERT-large scale, requires more data and compute
      12 layers: BERT-base scale, good balance for 33M cells pretraining
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1201   # 1200 genes + 1 <cls> token
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Stack of transformer blocks
        # nn.ModuleList is crucial here (not a plain list!):
        # It ensures all parameters are registered and moved with .to(device)
        self.layers = nn.ModuleList([
            scFMTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final LayerNorm (Pre-LN: apply after the last layer)
        self.final_norm = nn.LayerNorm(d_model)
        
        # Dropout on input embeddings
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights (very important for training stability)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Weight initialization is critical for transformer training.
        
        We use:
        - Linear layers: Normal(0, 0.02) — small std prevents initial
          saturation and keeps activations in a reasonable range
        - LayerNorm: weight=1, bias=0 (identity transform initially,
          model learns deviations from normalization)
        - Biases: Zero initialization (no initial bias)
        
        WHY 0.02?
          This is the standard choice from GPT and BERT papers.
          With depth=12 and residual connections, deeper layers receive
          accumulated variance. Some implementations scale the output
          projection of each layer by 1/sqrt(n_layers) to prevent
          activation variance from growing. scFM likely uses this
          "scaled initialization" but we use standard 0.02 here.
        
        PERFORMANCE TIP:
          For very deep models (n_layers > 24), use "scaled init":
          std = 0.02 / sqrt(2 * n_layers) for output projections.
          This keeps the residual stream variance stable at initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,           # [batch, seq_len, d_model] — embedding output
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_all_attn: bool = False,  # Save attention weights from all layers
        return_last_attn: bool = False  # Save only last layer (for GRN inference)
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass through all transformer blocks.
        
        Args:
            x: Input embeddings [batch, seq_len, d_model]
            attn_mask: scFM attention mask [seq_len, seq_len]
            key_padding_mask: Padding mask [batch, seq_len]
            return_all_attn: If True, collect attention weights from ALL layers
            return_last_attn: If True, collect only from last layer
        
        Returns:
            h: Final representations [batch, seq_len, d_model]
            attn_weights_list: List of attention tensors (or None)
        
        NOTE ON return_last_attn:
          For GRN inference (Fig. 6 in the paper), only the LAST layer's
          attention is used. The paper normalizes attention scores by row
          and column consecutively, then averages across the 8 heads.
          This reflects which genes most influence each other in the
          final representation.
        """
        # Apply input dropout
        x = self.dropout(x)
        
        attn_weights_list = [] if (return_all_attn or return_last_attn) else None
        
        for i, layer in enumerate(self.layers):
            # Only request attention weights when needed (saves memory/compute)
            need_attn = return_all_attn or (return_last_attn and i == self.n_layers - 1)
            x, attn_weights = layer(
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                return_attn_weights=need_attn
            )
            if need_attn and attn_weights is not None:
                attn_weights_list.append(attn_weights.detach().cpu())
        
        # Final normalization
        x = self.final_norm(x)
        
        return x, attn_weights_list


# =============================================================================
# SECTION 8: The Pretraining Prediction Head (Gene Expression Prediction)
# =============================================================================

class GeneExpressionPredictionHead(nn.Module):
    """
    The prediction head for the pretraining objective.
    
    Takes transformer output for each gene token and predicts
    its binned expression value.
    
    From the paper (Eq. 12): MSE loss over the unknown gene positions.
    
    Architecture: MLP with one hidden layer.
    
    WHY AN MLP HERE (not just a linear layer)?
      The transformer output for a gene at position j encodes complex
      contextual information from all other genes. A single linear layer
      might not have enough capacity to decode this into a scalar bin value.
      A 2-layer MLP allows a non-linear transformation.
      
    HIDDEN DIM CHOICE (d_model // 2 = 256):
      This is a "bottleneck" design: compress d_model → 256 → 1
      It forces the model to distill the most predictive features.
      Alternative: use d_model throughout (512 → 512 → 1)
      
    OUTPUT SIZE:
      Single scalar: the predicted bin value (continuous, not discrete)
      We predict a continuous value and use MSE, not cross-entropy.
      Why MSE over cross-entropy?
        - Expression bins have ordinal structure (bin 3 > bin 2 > bin 1)
        - MSE captures this ordinal relationship; CE treats all wrong bins equally
        - This is an important modeling decision that acknowledges bin ordering
    """
    
    def __init__(
        self,
        d_model: int = 512,
        hidden_dim: int = 256,
        n_bins: int = 51   # Paper uses 51 bins (0 = unexpressed, 1-50 = expressed)
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),  # Normalize before final projection
            nn.Linear(hidden_dim, 1)  # Predict a single continuous value
        )
        
        # PERFORMANCE NOTE ON LayerNorm IN MLP:
        #   Adding LayerNorm between hidden layers in the prediction head
        #   helps stabilize training, especially in the early stages when
        #   the transformer output embeddings are still large/variable.
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Transformer output for target positions [batch, n_target, d_model]
        Returns:
            pred: Predicted expression values [batch, n_target, 1]
        """
        return self.mlp(x).squeeze(-1)  # [batch, n_target]


# =============================================================================
# SECTION 9: Pretraining Loss Computation
# =============================================================================

def compute_gep_loss(
    predictions: torch.Tensor,    # [batch, seq_len] — model predictions
    targets: torch.Tensor,         # [batch, seq_len] — true binned values
    mask: torch.BoolTensor,        # [batch, seq_len] — True = compute loss here
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Gene Expression Prediction (GEP) loss.
    
    From paper Eq. 12:
        L = (1/|U_unk|) * sum_{j in U_unk} (MLP(h_n^i) - x_j^i)^2
    
    This is masked MSE: only compute loss at the masked/unknown positions.
    
    DESIGN CHOICES:
    
    1. Why not predict all genes?
       Computing loss on ALL genes would include many zero-expression genes.
       Zeros dominate expression data (sparsity ~80-95%).
       This would bias the model toward predicting zeros everywhere.
       
    2. Masking strategy during fine-tuning (GEP):
       - Random 40% of genes are masked (mask_ratio = 0.4, from paper Methods)
       - This is different from pretraining where a variable proportion
         (25%, 50%, or 75% uniformly sampled) is used
       - 40% for fine-tuning provides sufficient signal without over-masking
    
    3. Prediction target: BINNED values (0-50) not raw counts
       Bins are the cleaner prediction target because they're batch-corrected.
       
    PERFORMANCE TIP:
       For a large vocabulary of bins, consider also using cross-entropy
       loss with ordinal encoding as an additional signal. In practice,
       MSE tends to work well because the ordinal structure helps.
    """
    # Get predictions and targets at masked positions
    masked_pred = predictions[mask]    # [n_masked_total]
    masked_target = targets[mask].float()  # [n_masked_total]
    
    if masked_pred.numel() == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    loss = F.mse_loss(masked_pred, masked_target, reduction=reduction)
    return loss


# =============================================================================
# SECTION 10: Complete scFM Model (Embedding + Transformer + Head)
# =============================================================================

class scFMModel(nn.Module):
    """
    The complete scFM model integrating all components.
    
    This class combines:
    1. Input embeddings (Module 1): gene + expression + condition embeddings
    2. Transformer (Module 2): 12-layer transformer with masked attention
    3. Prediction head: MLP for expression value prediction
    
    Usage modes:
    - Pretraining: predict masked gene expression (GEP objective)
    - Fine-tuning cell type: extract <cls> embedding → classifier
    - Fine-tuning batch integration: extract <cls> embedding → cluster
    - GRN inference: extract attention maps from last layer
    """
    
    def __init__(
        self,
        vocab_size: int = 20_001,    # ~20K human genes + special tokens
        n_conditions: int = 10,      # Number of batch/condition categories
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1201,
        n_bins: int = 51,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # --- Input Embeddings (from Module 1) ---
        # Gene token embedding: maps integer gene ID → d_model vector
        self.gene_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Expression value embedding: maps continuous bin value → d_model vector
        # Using Linear (not Embedding) to preserve ordinal relationships
        self.expr_embedding = nn.Linear(1, d_model, bias=True)
        
        # Condition/batch embedding: maps condition ID → d_model vector
        self.condition_embedding = nn.Embedding(n_conditions, d_model)
        
        # Layer normalization and dropout on combined embedding
        self.embedding_norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # --- Transformer ---
        self.transformer = scFMTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        # --- Prediction Head ---
        self.expr_pred_head = GeneExpressionPredictionHead(
            d_model=d_model,
            hidden_dim=d_model // 2,
            n_bins=n_bins
        )
        
        # Initialize embeddings with small std for stable training
        nn.init.normal_(self.gene_embedding.weight, std=0.02)
        nn.init.normal_(self.condition_embedding.weight, std=0.02)
    
    def get_embeddings(
        self,
        gene_tokens: torch.Tensor,       # [batch, seq_len] — integer gene IDs
        expr_values: torch.Tensor,        # [batch, seq_len] — float bin values
        condition_tokens: torch.Tensor    # [batch, seq_len] — integer condition IDs
    ) -> torch.Tensor:
        """
        Compute combined three-part embedding (Eq. 5 in paper).
        
        h = emb_g(t_g) + emb_x(x) + emb_c(t_c)
        
        Element-wise SUM (not concatenation):
        - Sum keeps the representation in d_model dimensions
        - Each embedding type contributes equally (before training)
        - The model learns to weight them during training
        - Concatenation would triple the dimension (3×512=1536),
          requiring larger downstream layers
        """
        gene_emb = self.gene_embedding(gene_tokens)  # [batch, seq_len, d_model]
        
        # Expression embedding: reshape for Linear layer
        expr_emb = self.expr_embedding(
            expr_values.float().unsqueeze(-1)  # [batch, seq_len, 1]
        )  # [batch, seq_len, d_model]
        
        cond_emb = self.condition_embedding(condition_tokens)  # [batch, seq_len, d_model]
        
        # Element-wise sum + normalize
        h = gene_emb + expr_emb + cond_emb  # [batch, seq_len, d_model]
        h = self.embedding_norm(h)
        h = self.embedding_dropout(h)
        
        return h
    
    def forward(
        self,
        gene_tokens: torch.Tensor,           # [batch, seq_len]
        expr_values: torch.Tensor,            # [batch, seq_len]
        condition_tokens: torch.Tensor,       # [batch, seq_len]
        attn_mask: Optional[torch.Tensor] = None,     # [seq_len, seq_len]
        key_padding_mask: Optional[torch.Tensor] = None,  # [batch, seq_len]
        return_cell_embedding: bool = False,
        return_attn_weights: bool = False
    ) -> dict:
        """
        Full forward pass.
        
        Returns a dict with:
          'gene_repr':    Gene-level representations [batch, seq_len, d_model]
          'cell_repr':    Cell-level representation [batch, d_model] (from <cls>)
          'expr_pred':    Expression predictions [batch, seq_len] (from pred head)
          'attn_weights': List of attention weight tensors (if requested)
        """
        # 1. Compute embeddings
        h = self.get_embeddings(gene_tokens, expr_values, condition_tokens)
        
        # 2. Run through transformer
        h, attn_weights_list = self.transformer(
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            return_last_attn=return_attn_weights
        )
        # h: [batch, seq_len, d_model]
        
        # 3. Extract outputs
        result = {}
        
        # Gene-level representations (all positions)
        result['gene_repr'] = h  # [batch, seq_len, d_model]
        
        # Cell-level representation from <cls> token (position 0)
        result['cell_repr'] = h[:, 0, :]  # [batch, d_model]
        
        # Expression predictions (for all positions, loss computed at masked ones)
        result['expr_pred'] = self.expr_pred_head(h)  # [batch, seq_len]
        
        if return_attn_weights:
            result['attn_weights'] = attn_weights_list
        
        return result


# =============================================================================
# SECTION 11: Demonstrations
# =============================================================================

def demo_attention_mask():
    demonstrate_attention_mask()


def demo_transformer_block():
    """
    Test a single transformer block with a small synthetic input.
    """
    print("=" * 60)
    print("Transformer Block Demo")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 10
    d_model = 512
    
    block = scFMTransformerBlock(d_model=d_model, n_heads=8, d_ff=512)
    
    # Random input (simulates embedding output)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create a simple attention mask: last 3 positions are "unknown"
    known_mask = torch.tensor([True]*7 + [False]*3)
    attn_mask = build_scgpt_attention_mask_vectorized(seq_len, known_mask, device='cpu')
    
    # Forward pass (no gradients needed for demo)
    with torch.no_grad():
        output, attn_weights = block(x, attn_mask=attn_mask, return_attn_weights=True)
    
    print(f"Input shape:   {x.shape}")
    print(f"Output shape:  {output.shape}")
    print(f"Attn weights:  {attn_weights.shape}")  # [batch, n_heads, seq_len, seq_len]
    
    # Verify output shape matches input (residual connection preserves shape)
    assert output.shape == x.shape, "Output shape mismatch!"
    print(f"\n✓ Output shape equals input shape (residual connections working)")
    
    # Verify masking: attention from unknown to unknown should be ~0
    # Position 7 (unknown) attending to position 8 (unknown) should be ~0
    attn_from_pos7_to_pos8 = attn_weights[0, :, 7, 8].mean()  # avg over heads
    print(f"✓ Attn from unknown→unknown (should ≈0): {attn_from_pos7_to_pos8:.6f}")
    
    # Verify: attention from unknown to known should be nonzero
    attn_from_pos7_to_pos0 = attn_weights[0, :, 7, 0].mean()  # pos 7 → pos 0 (known)
    print(f"✓ Attn from unknown→known (should >0):  {attn_from_pos7_to_pos0:.6f}")
    print()


def demo_full_model():
    """
    End-to-end forward pass through the complete scFM model.
    Shows shapes at each step and memory requirements.
    """
    print("=" * 60)
    print("Full scFM Model Demo")
    print("=" * 60)
    
    # Use smaller dimensions for demo
    batch_size = 4
    seq_len = 50    # 49 genes + 1 <cls>
    vocab_size = 500
    
    model = scFMModel(
        vocab_size=vocab_size,
        n_conditions=5,
        d_model=128,  # Smaller for demo
        n_heads=4,
        n_layers=3,   # Fewer layers for demo
        d_ff=128,
        max_seq_len=seq_len + 1,
        n_bins=51
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Create synthetic batch
    # <cls> token at position 0, genes at positions 1..seq_len-1
    cls_token_id = vocab_size - 1  # Use last vocab id as <cls>
    gene_tokens = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
    gene_tokens[:, 0] = cls_token_id  # First token is always <cls>
    
    # Expression values: 0 for <cls>, random bins 1-50 for genes
    expr_values = torch.randint(0, 51, (batch_size, seq_len)).float()
    expr_values[:, 0] = 0.0  # <cls> has no expression value
    
    condition_tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)  # All batch 0
    
    # Create masks
    # Last 10 genes are "unknown" (to be predicted)
    n_unknown = 10
    known_mask = torch.tensor([True] * (seq_len - n_unknown) + [False] * n_unknown)
    attn_mask = build_scgpt_attention_mask_vectorized(seq_len, known_mask, device='cpu')
    
    # Padding mask: some cells have fewer genes (padded with token 0)
    key_padding_mask = (gene_tokens == 0)  # True = padding position
    
    print(f"\nInput shapes:")
    print(f"  gene_tokens:      {gene_tokens.shape}")
    print(f"  expr_values:      {expr_values.shape}")
    print(f"  condition_tokens: {condition_tokens.shape}")
    print(f"  attn_mask:        {attn_mask.shape}")
    print(f"  padding_mask:     {key_padding_mask.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(
            gene_tokens, expr_values, condition_tokens,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            return_cell_embedding=True,
            return_attn_weights=True
        )
    
    print(f"\nOutput shapes:")
    print(f"  gene_repr:    {output['gene_repr'].shape}")
    print(f"  cell_repr:    {output['cell_repr'].shape}")
    print(f"  expr_pred:    {output['expr_pred'].shape}")
    print(f"  attn_weights: {len(output['attn_weights'])} layer(s), each {output['attn_weights'][0].shape}")
    
    # Compute pretraining loss
    target_mask = ~known_mask  # Compute loss at unknown positions
    target_mask_expanded = target_mask.unsqueeze(0).expand(batch_size, -1)
    
    loss = compute_gep_loss(
        output['expr_pred'],
        expr_values,
        target_mask_expanded
    )
    print(f"\nPretraining GEP loss: {loss.item():.4f}")
    print(f"(MSE between predicted and true bin values at {n_unknown} unknown positions)")
    
    # Memory estimation
    # Activations during training (rough estimate)
    bytes_per_float = 4  # float32
    activation_mem = batch_size * seq_len * 128 * 3 * bytes_per_float  # ~3 per layer
    print(f"\nEstimated activation memory (mini example): {activation_mem / 1024:.1f} KB")
    print(f"At production scale (batch=32, seq=1200, d=512): ~{32*1200*512*3*4/1024/1024:.0f} MB per layer")
    print()


def demo_training_step():
    """
    Shows a complete training step: forward, loss, backward, optimizer update.
    This is the core loop you'll repeat millions of times during pretraining.
    """
    print("=" * 60)
    print("Training Step Demo")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 20
    vocab_size = 200
    
    model = scFMModel(
        vocab_size=vocab_size, n_conditions=3, d_model=64,
        n_heads=2, n_layers=2, d_ff=64, n_bins=51
    )
    
    # Adam optimizer — standard for transformers
    # lr=1e-4 from paper; weight_decay acts as L2 regularization
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,          # Learning rate from paper
        weight_decay=0.9  # NOTE: This is per-epoch decay (unusual), not standard WD
        # Standard weight_decay would be 0.01-0.1; the paper's 0.9 is
        # a per-epoch learning rate schedule factor, not L2 regularization.
        # In practice, use: weight_decay=0.01 and a separate LR scheduler.
    )
    
    # Create a learning rate scheduler (cosine annealing is common)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=6  # 6 epochs from paper
    )
    
    # Simulated batch
    gene_tokens = torch.randint(1, vocab_size-1, (batch_size, seq_len))
    gene_tokens[:, 0] = vocab_size - 1  # <cls>
    expr_values = torch.randint(0, 51, (batch_size, seq_len)).float()
    expr_values[:, 0] = 0.0
    condition_tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    # Create random mask: 50% genes are unknown this step
    # During pretraining: randomly sample from {0.25, 0.50, 0.75}
    mask_ratio = 0.50
    n_genes = seq_len - 1  # Exclude <cls>
    n_unknown = int(n_genes * mask_ratio)
    
    # Random permutation to select unknown positions
    gene_indices = torch.randperm(n_genes) + 1  # +1 to skip <cls> at pos 0
    unknown_positions = gene_indices[:n_unknown]
    
    known_mask = torch.ones(seq_len, dtype=torch.bool)
    known_mask[unknown_positions] = False
    
    # For unknown genes: set expression values to 0 in input
    # (model doesn't get to see the target values!)
    input_expr = expr_values.clone()
    for pos in unknown_positions:
        input_expr[:, pos] = 0.0
    
    attn_mask = build_scgpt_attention_mask_vectorized(seq_len, known_mask, device='cpu')
    
    print(f"Mask ratio: {mask_ratio} ({n_unknown}/{n_genes} genes masked)")
    
    # Forward pass
    model.train()
    output = model(gene_tokens, input_expr, condition_tokens, attn_mask=attn_mask)
    
    # Compute loss at unknown positions
    target_mask = ~known_mask
    target_mask[0] = False  # Never compute loss at <cls> position
    target_mask_expanded = target_mask.unsqueeze(0).expand(batch_size, -1)
    
    loss = compute_gep_loss(output['expr_pred'], expr_values, target_mask_expanded)
    
    print(f"Forward pass complete, loss = {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping — very important for transformers!
    # WHY CLIP GRADIENTS?
    #   Transformers can produce large gradient norms early in training,
    #   causing "catastrophic" parameter updates that destroy learning.
    #   Clipping to max_norm=1.0 limits this while preserving direction.
    #   Without clipping, you'll frequently see training instability (loss NaN).
    max_norm = 1.0
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    print(f"Gradient norm (clipped to {max_norm}): {grad_norm:.4f}")
    
    optimizer.step()
    print(f"Optimizer step complete, lr = {scheduler.get_last_lr() if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']:.2e}")
    print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("scFM Tutorial — Module 2: Masked Attention Transformer")
    print("="*70 + "\n")
    
    print("--- Demo 1: Attention Mask Visualization ---\n")
    demo_attention_mask()
    
    print("\n--- Demo 2: Transformer Block ---\n")
    demo_transformer_block()
    
    print("--- Demo 3: Full Model Forward Pass ---\n")
    demo_full_model()
    
    print("--- Demo 4: Complete Training Step ---\n")
    demo_training_step()
    
    print("=" * 60)
    print("Module 2 Complete!")
    print("=" * 60)
    print("""
    Key concepts covered:
    
    1. scFM attention mask: known genes attend freely; unknown genes
       can only attend to known genes + themselves. This enables
       generative pretraining on non-sequential gene sets.
    
    2. Multi-head attention (8 heads × 64 dims): each head learns
       different aspects of gene-gene relationships in parallel.
    
    3. Pre-LayerNorm design: normalizes BEFORE each sublayer for
       training stability in deep (12-layer) networks.
    
    4. <cls> token: a special "readout" token that aggregates all
       gene information into a single cell-level representation.
    
    5. GEP loss: MSE at masked positions only. Gradient clipping
       (max_norm=1.0) is critical for stable transformer training.
    
    Next: Module 3 will cover fine-tuning objectives (GEP, GEPC, ECS, DAR)
    and cell type classification.
    """)
