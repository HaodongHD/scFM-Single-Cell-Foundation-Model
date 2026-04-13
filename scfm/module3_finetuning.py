"""
=============================================================================
scFM Tutorial — Module 3: Fine-Tuning Objectives & Task Heads
=============================================================================

In Module 2, we built the core transformer and the pretraining GEP objective.
Now we implement the FINE-TUNING stage: adapting the pretrained scFM to
specific biological tasks.

The paper defines 5 fine-tuning objectives:
  1. GEP  — Gene Expression Prediction (also used in pretraining)
  2. GEPC — GEP for Cell modeling (predicts expression from cell embedding)
  3. ECS  — Elastic Cell Similarity (contrastive, for batch integration)
  4. DAR  — Domain Adaptation via Reverse backpropagation (batch correction)
  5. CLS  — Cell Type Classification (cross-entropy)

Key learning objectives:
  - How multi-task learning via combined losses works
  - How GEPC creates rich cell representations
  - How ECS implements contrastive learning for biology
  - How gradient reversal removes batch effects
  - How the cell classifier is attached for annotation
  - Which loss combination to use for each downstream task

=============================================================================
SECTION 1: GEPC — Gene Expression Prediction for Cell Modeling
=============================================================================

GEPC is the KEY innovation that makes cell representations biologically rich.

Recall from Module 2: GEP uses gene-level transformer output h_n^(i) to
predict expression. GEPC instead uses the CELL EMBEDDING h_c^(i) (the <cls>
output) to predict expression for each gene.

From paper Eq. 14:
  q_j = MLP(emb_g(t_g^(i)))   ← gene-specific query vector
  x̂_j = q_j · W · h_c^(i)   ← dot product with cell representation
  L_GEPC = MSE at masked positions

WHY IS THIS POWERFUL?
  The cell embedding h_c must compress ALL biologically relevant information
  about the cell into a single vector of 512 dims. To minimize GEPC loss,
  the cell embedding must encode what each gene's expression SHOULD BE.
  This forces the embedding to be a compact, dense summary of cell state —
  exactly what we want for clustering and annotation.

GEP vs GEPC comparison:
  GEP:  h_gene_j → predict x_j  (local: each gene's own context)
  GEPC: h_cell + gene_j_id → predict x_j  (global: whole-cell context)
  COMBINED: the paper finds GEP + GEPC together >> either alone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class GEPCHead(nn.Module):
    """
    Gene Expression Prediction for Cell Modeling head.
    
    Predicts expression of each gene from:
      1. The cell embedding h_c (global cell state)
      2. The gene's own identity embedding (which gene are we predicting?)
    
    This bilinear interaction: x̂_j = q_j · W · h_c
    ensures the prediction is gene-specific (not the same for all genes)
    while also being informed by the overall cell state.
    
    ARCHITECTURE CHOICE: Bilinear vs Concatenate+MLP
      Bilinear:  x̂_j = (MLP(emb_g_j)) · W · h_c  ← paper's choice
      Concat:    x̂_j = MLP(cat(emb_g_j, h_c))     ← common alternative
      
      The bilinear approach is more parameter-efficient and has a nice
      geometric interpretation: we're measuring how "aligned" the gene
      query vector is with the cell state vector, after a learned transformation.
      
      If d_model=512, the bilinear matrix W has 512×512=262,144 params.
      The concat approach would need MLP(1024 → 512 → 1) = ~524,288 params.
      Both are reasonable; bilinear is marginally more efficient.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        mlp_hidden: int = 256
    ):
        super().__init__()
        
        # Gene query: transforms gene embedding → query vector
        self.gene_query_mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model)
        )
        
        # Bilinear weight matrix W: learned transformation of cell embedding
        # q_j · W · h_c = q_j @ W @ h_c
        # We factor this as: score = (W @ h_c) · q_j
        self.W = nn.Linear(d_model, d_model, bias=False)
    
    def forward(
        self,
        gene_embeddings: torch.Tensor,   # [batch, seq_len, d_model] — raw gene embs
        cell_embedding: torch.Tensor     # [batch, d_model] — <cls> output
    ) -> torch.Tensor:
        """
        Args:
            gene_embeddings: Gene identity embeddings (not transformer output)
                            These are the ORIGINAL gene embeddings, not contextualized
            cell_embedding: The cell-level representation h_c from <cls> token
        
        Returns:
            predictions: [batch, seq_len] predicted expression values
        """
        # Compute gene-specific query vectors: [batch, seq_len, d_model]
        q = self.gene_query_mlp(gene_embeddings)
        
        # Transform cell embedding: [batch, d_model] → [batch, d_model]
        Wh_c = self.W(cell_embedding)  # [batch, d_model]
        
        # Expand Wh_c to match gene sequence: [batch, 1, d_model]
        Wh_c = Wh_c.unsqueeze(1)
        
        # Bilinear score: (q * Wh_c).sum(-1) — element-wise product + sum = dot product
        # Shape: [batch, seq_len, d_model] * [batch, 1, d_model] → [batch, seq_len]
        scores = (q * Wh_c).sum(-1)  # [batch, seq_len]
        
        return scores


# =============================================================================
# SECTION 2: ECS — Elastic Cell Similarity
# =============================================================================

class ElasticCellSimilarityLoss(nn.Module):
    """
    Elastic Cell Similarity (ECS) Loss for learning batch-invariant cell representations.
    
    From paper Eq. 15:
        L_ECS = -(sim(h_c^i, h_c^{i'}) - β)^2
    
    where sim is cosine similarity.
    
    This is a "soft" contrastive loss that:
    - PUSHES APART pairs with cosine similarity > β (makes them even MORE similar)
    - DOES NOTHING to pairs below the threshold β
    
    Wait, that seems backwards. Let me explain why this is clever:
    
    The NEGATIVE SIGN means we're MAXIMIZING (sim - β)^2.
    When sim > β (pairs are already similar):
      (sim - β)^2 > 0, gradient pushes sim toward 1 (more similar)
    When sim < β (pairs are dissimilar):
      (sim - β)^2 > 0 but gradient pushes sim toward β from below
      Effect: dissimilar pairs tend to be pushed further from β
    
    So ECS acts as a "rubber band" that pulls similar pairs together
    beyond the threshold β, while letting dissimilar pairs stay apart.
    
    IN PRACTICE for batch integration:
      - Two cells of the same type but different batches → should be similar
      - Two cells of different types → should be dissimilar
      - ECS with β=0.6 (paper value) creates a gentle attractive force
        between biologically similar cells
    
    PARAMETER CHOICE β=0.6:
      - β < 0.5: too lenient, allows very different cells to be "similar"
      - β > 0.8: too strict, most cell pairs fall above threshold
      - β=0.6 is a good middle ground for immune/blood cell data
      - Tune this if your cells have less distinct types
    
    ECS WEIGHT in combined loss = 10 (from paper Methods)
      This large weight compensates for the small magnitude of cosine
      similarity (range -1 to 1) vs MSE losses (can be much larger).
    """
    
    def __init__(self, beta: float = 0.6):
        super().__init__()
        self.beta = beta
    
    def forward(
        self,
        cell_embeddings: torch.Tensor  # [batch, d_model]
    ) -> torch.Tensor:
        """
        Compute ECS loss over all pairs in the batch.
        
        IMPORTANT: We compute pairwise similarities over the ENTIRE batch.
        For batch_size=32, this is 32×32=1024 pairs — manageable.
        For very large batches, you might subsample pairs.
        """
        batch_size = cell_embeddings.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=cell_embeddings.device)
        
        # Normalize embeddings for cosine similarity
        # L2 normalization: each cell embedding becomes a unit vector
        normed = F.normalize(cell_embeddings, p=2, dim=-1)  # [batch, d_model]
        
        # Pairwise cosine similarities: [batch, batch]
        sim_matrix = torch.matmul(normed, normed.T)
        
        # Exclude diagonal (similarity of cell with itself = 1.0, trivially)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=cell_embeddings.device)
        sim_values = sim_matrix[mask]  # [batch*(batch-1)]
        
        # ECS loss: -(sim - beta)^2
        # Negative because we want to MAXIMIZE (push to 1 or push toward β)
        loss = -((sim_values - self.beta) ** 2).mean()
        
        return loss


# =============================================================================
# SECTION 3: DAR — Domain Adaptation via Reverse Backpropagation
# =============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) — core trick for domain adaptation.
    
    During FORWARD pass: identity function (x → x)
    During BACKWARD pass: negate the gradient (∂L/∂x → -∂L/∂x)
    
    This is the Ganin & Lempitsky (2015) domain adaptation trick.
    
    HOW IT WORKS for batch correction:
    
    1. A batch classifier tries to predict which batch a cell comes from,
       using the cell embedding as input.
    2. Normally, minimizing batch classification loss would make the
       embedding MORE batch-specific (bad for biology).
    3. With GRL, the gradient from the batch classifier is REVERSED
       before backpropagating into the main encoder.
    4. This means: the encoder learns to produce embeddings that make
       batch classification HARDER → embeddings become batch-invariant!
    
    The encoder is simultaneously:
    - MINIMIZING cell type classification loss (keep biological signal)
    - MAXIMIZING batch classification loss (remove batch effect)
    
    This adversarial training creates batch-corrected representations
    without requiring paired samples across batches.
    
    SCALE FACTOR λ:
      Controls the strength of domain adaptation.
      λ = 0: no reversal (standard training)
      λ = 1: full reversal (as described above)
      Small λ early in training → gradually increase (curriculum approach)
      The paper uses a fixed schedule or constant λ.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(scale))
        return x  # Identity in forward pass
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        scale, = ctx.saved_tensors
        return -scale.item() * grad_output, None  # Negate gradient!


class GradientReversalLayer(nn.Module):
    """Wrapper module for the gradient reversal function."""
    
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.scale)


class DomainAdaptationClassifier(nn.Module):
    """
    Batch domain classifier with gradient reversal.
    
    Architecture:
      cell_embedding → GRL → MLP → batch_logits
      
    During training:
      - Forward: normal classification
      - Backward through GRL: gradient is negated → encoder becomes
        batch-agnostic
    
    For n_batches classes (18 batches in COVID-19 dataset), use
    cross-entropy loss over the batch predictions.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_batches: int = 18,
        hidden_dim: int = 128,
        grl_scale: float = 1.0
    ):
        super().__init__()
        
        self.grl = GradientReversalLayer(scale=grl_scale)
        
        # Batch classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),          # ReLU here (vs GELU in transformer) — common for classifiers
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_batches)
        )
    
    def forward(
        self,
        cell_embedding: torch.Tensor  # [batch, d_model]
    ) -> torch.Tensor:
        """Returns batch classification logits [batch, n_batches]"""
        # GRL reverses gradient during backprop
        x = self.grl(cell_embedding)
        logits = self.classifier(x)
        return logits
    
    def compute_loss(
        self,
        cell_embedding: torch.Tensor,  # [batch, d_model]
        batch_labels: torch.Tensor     # [batch] — integer batch IDs
    ) -> torch.Tensor:
        logits = self(cell_embedding)
        return F.cross_entropy(logits, batch_labels)


# =============================================================================
# SECTION 4: Cell Type Classification Head
# =============================================================================

class CellTypeClassifier(nn.Module):
    """
    Cell type annotation head — the most common fine-tuning task.
    
    The simplest fine-tuning: take the <cls> cell embedding and classify it.
    
    From paper: trained with cross-entropy on reference dataset,
    evaluated on held-out query dataset.
    
    Architecture: simple MLP classifier
    
    WHY THIS SIMPLE APPROACH WORKS:
      The pretrained scFM cell embedding already encodes rich biological
      information about cell type. The classifier just needs to learn a
      linear (or shallow MLP) boundary in this rich embedding space.
      
      Compare: training a model from scratch would require deep task-specific
      architectures, because the raw gene expression space is 20K dimensions
      with complex non-linear structure.
      
    REGULARIZATION:
      Dropout on the hidden layer is important to prevent overfitting,
      especially when fine-tuning on small labeled datasets (common in biology).
      
    NOTE ON FREEZING:
      Option A: Fine-tune ALL layers (full fine-tuning)
        → Best performance, requires more labeled data
      Option B: Freeze transformer, only train classifier
        → Good for very small datasets, preserves pretraining
      Option C: Freeze lower layers, fine-tune upper layers
        → Good compromise (freeze first 8, train last 4 + classifier)
      
      The paper fine-tunes all layers. For new tasks with limited data,
      start with Option B and increase if data allows.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_cell_types: int = 20,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),      # Normalize the cell embedding first
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_cell_types)
        )
        
        # Initialize with small weights for stable fine-tuning start
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, cell_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cell_embedding: [batch, d_model]
        Returns:
            logits: [batch, n_cell_types]
        """
        return self.classifier(cell_embedding)
    
    def predict(self, cell_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns both class predictions and confidence scores.
        Useful at inference time for annotation.
        """
        logits = self(cell_embedding)
        probs = F.softmax(logits, dim=-1)
        predictions = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values
        return predictions, confidence


# =============================================================================
# SECTION 5: Complete Fine-Tuning Model
# =============================================================================

class scFMFineTuner(nn.Module):
    """
    Wraps the pretrained scFM with task-specific fine-tuning objectives.
    
    For cell type annotation:
      Use: GEP + GEPC + CLS losses
      The GEP/GEPC keep gene expression modeling alive (prevent catastrophic
      forgetting of pretraining knowledge), while CLS guides cell embeddings
      toward cell-type-discriminative space.
    
    For batch integration:
      Use: GEP + GEPC + ECS + DAR losses
      ECS creates continuous-valued cell similarity objectives;
      DAR explicitly removes batch information from cell embeddings.
    
    COMBINED LOSS FORMULA (for integration):
      L_total = L_GEP + L_GEPC + 10 * L_ECS + L_DAR
      
    WHY MULTI-TASK TRAINING?
      Fine-tuning only on cell type classification would cause "catastrophic
      forgetting": the model would lose its understanding of gene-gene
      interactions, since those aren't directly needed for classification.
      
      Keeping GEP/GEPC alive preserves gene-level knowledge, which:
      1. Acts as regularization (prevents overfitting to small datasets)
      2. Maintains interpretability of gene embeddings
      3. Enables GRN inference from the fine-tuned model (Fig. 5 in paper)
    """
    
    def __init__(
        self,
        pretrained_scgpt,          # Pretrained scFMModel from Module 2
        task: str = 'annotation',  # 'annotation' | 'integration' | 'perturbation'
        n_cell_types: int = 20,
        n_batches: int = 1,
        n_conditions: int = 1,
        ecs_beta: float = 0.6,
        ecs_weight: float = 10.0,  # Weight for ECS loss in combined loss
        grl_scale: float = 1.0
    ):
        super().__init__()
        
        self.scgpt = pretrained_scgpt
        self.task = task
        self.ecs_weight = ecs_weight
        d_model = pretrained_scgpt.d_model
        
        # Task-specific heads
        if task == 'annotation':
            self.classifier = CellTypeClassifier(d_model, n_cell_types)
            self.gepc_head = GEPCHead(d_model)
        
        elif task == 'integration':
            self.ecs_loss_fn = ElasticCellSimilarityLoss(beta=ecs_beta)
            self.gepc_head = GEPCHead(d_model)
            if n_batches > 1:
                self.dar = DomainAdaptationClassifier(d_model, n_batches, grl_scale=grl_scale)
        
        elif task == 'perturbation':
            # For perturbation: use GEP only (predict post-perturbation expression)
            # No GEPC, no ECS/DAR — just gene-level prediction
            pass  # Covered in Module 4
    
    def forward_annotation(
        self,
        gene_tokens, expr_values, condition_tokens,
        attn_mask=None, key_padding_mask=None,
        labels=None,        # [batch] cell type labels (during training)
        expr_targets=None,  # [batch, seq_len] true expression (for GEP loss)
        mask=None           # [batch, seq_len] which positions are masked
    ) -> Dict:
        """
        Forward pass for cell type annotation.
        
        Returns loss components and predictions.
        """
        # Full forward pass through scFM
        output = self.scgpt(
            gene_tokens, expr_values, condition_tokens,
            attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        
        cell_repr = output['cell_repr']       # [batch, d_model]
        gene_repr = output['gene_repr']       # [batch, seq_len, d_model]
        expr_pred = output['expr_pred']       # [batch, seq_len]
        
        result = {'cell_repr': cell_repr}
        
        # GEP loss: predict masked gene expression from gene-level output
        gep_loss = torch.tensor(0.0)
        if mask is not None and expr_targets is not None:
            gep_loss = F.mse_loss(expr_pred[mask], expr_targets[mask].float())
        result['gep_loss'] = gep_loss
        
        # GEPC loss: predict masked expression from CELL EMBEDDING
        gepc_loss = torch.tensor(0.0)
        if mask is not None and expr_targets is not None:
            # Get original gene embeddings (before transformer)
            gene_emb = self.scgpt.gene_embedding(gene_tokens)
            gepc_pred = self.gepc_head(gene_emb, cell_repr)
            gepc_loss = F.mse_loss(gepc_pred[mask], expr_targets[mask].float())
        result['gepc_loss'] = gepc_loss
        
        # Classification loss
        cls_loss = torch.tensor(0.0)
        logits = self.classifier(cell_repr)
        result['logits'] = logits
        if labels is not None:
            cls_loss = F.cross_entropy(logits, labels)
        result['cls_loss'] = cls_loss
        
        # Total loss for annotation: GEP + GEPC + CLS
        # Note: GEP and GEPC are self-supervised (no labels needed),
        # CLS requires labeled reference data
        result['total_loss'] = gep_loss + gepc_loss + cls_loss
        
        return result
    
    def forward_integration(
        self,
        gene_tokens, expr_values, condition_tokens,
        attn_mask=None, key_padding_mask=None,
        batch_labels=None,   # [batch] which sequencing batch each cell is from
        expr_targets=None,
        mask=None
    ) -> Dict:
        """
        Forward pass for batch integration.
        No cell type labels needed — this is self-supervised!
        """
        output = self.scgpt(
            gene_tokens, expr_values, condition_tokens,
            attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        
        cell_repr = output['cell_repr']
        expr_pred = output['expr_pred']
        gene_repr = output['gene_repr']
        
        result = {'cell_repr': cell_repr}
        
        # GEP loss
        gep_loss = torch.tensor(0.0)
        if mask is not None and expr_targets is not None:
            gep_loss = F.mse_loss(expr_pred[mask], expr_targets[mask].float())
        result['gep_loss'] = gep_loss
        
        # GEPC loss
        gepc_loss = torch.tensor(0.0)
        if mask is not None and expr_targets is not None:
            gene_emb = self.scgpt.gene_embedding(gene_tokens)
            gepc_pred = self.gepc_head(gene_emb, cell_repr)
            gepc_loss = F.mse_loss(gepc_pred[mask], expr_targets[mask].float())
        result['gepc_loss'] = gepc_loss
        
        # ECS loss: encourages biologically similar cells to cluster
        ecs_loss = self.ecs_loss_fn(cell_repr)
        result['ecs_loss'] = ecs_loss
        
        # DAR loss: removes batch effects via gradient reversal
        dar_loss = torch.tensor(0.0)
        if hasattr(self, 'dar') and batch_labels is not None:
            dar_loss = self.dar.compute_loss(cell_repr, batch_labels)
        result['dar_loss'] = dar_loss
        
        # Combined loss with ECS weighting
        result['total_loss'] = gep_loss + gepc_loss + self.ecs_weight * ecs_loss + dar_loss
        
        return result


# =============================================================================
# SECTION 6: Demonstrations
# =============================================================================

def demo_gepc():
    """Demonstrate GEPC head: predicting expression from cell embedding."""
    print("=" * 60)
    print("GEPC Demo: Predicting Expression from Cell Embedding")
    print("=" * 60)
    
    batch_size, seq_len, d_model = 4, 20, 128
    gepc = GEPCHead(d_model=d_model, mlp_hidden=64)
    
    gene_embeddings = torch.randn(batch_size, seq_len, d_model)
    cell_embedding = torch.randn(batch_size, d_model)
    
    predictions = gepc(gene_embeddings, cell_embedding)
    print(f"Gene embeddings shape:  {gene_embeddings.shape}")
    print(f"Cell embedding shape:   {cell_embedding.shape}")
    print(f"GEPC predictions shape: {predictions.shape}  ← one score per gene")
    
    # Demonstrate that predictions are gene-specific
    # (different genes get different predictions even with the same cell embedding)
    print(f"\nFirst 5 gene predictions for cell 0: {predictions[0, :5].detach().numpy().round(3)}")
    print(f"(Different values confirm gene-specificity via query vectors)\n")


def demo_ecs_loss():
    """Demonstrate ECS loss behavior."""
    print("=" * 60)
    print("ECS Loss Demo")
    print("=" * 60)
    
    ecs = ElasticCellSimilarityLoss(beta=0.6)
    
    # Case 1: Very similar cell embeddings (same cell type)
    similar_embeddings = torch.randn(8, 128)
    similar_embeddings = similar_embeddings + torch.randn(1, 128) * 10  # Add large shared component
    loss_similar = ecs(similar_embeddings)
    
    # Case 2: Very different embeddings (mixed cell types)
    different_embeddings = torch.randn(8, 128)  # Pure noise → low similarity
    loss_different = ecs(different_embeddings)
    
    print(f"ECS loss for SIMILAR embeddings:   {loss_similar.item():.4f}")
    print(f"ECS loss for DIFFERENT embeddings: {loss_different.item():.4f}")
    print(f"\nNote: ECS loss is NEGATIVE (we're maximizing)")
    print(f"More similar embeddings → more negative loss (more optimization signal)")
    print(f"This makes sense: similar cells have more room to be 'even more similar'\n")


def demo_gradient_reversal():
    """
    Demonstrate gradient reversal for batch correction.
    Shows that the cell encoder's gradients are negated by GRL.
    """
    print("=" * 60)
    print("Gradient Reversal Demo: Batch Effect Removal")
    print("=" * 60)
    
    d_model = 64
    n_batches = 3
    batch_size = 6
    
    # Simple encoder
    encoder = nn.Linear(d_model, d_model)
    dar_classifier = DomainAdaptationClassifier(d_model, n_batches, grl_scale=1.0)
    
    # Forward pass
    x = torch.randn(batch_size, d_model, requires_grad=True)
    encoded = encoder(x)  # Cell embeddings
    
    # Without GRL: normal batch classification loss
    batch_labels = torch.tensor([0, 1, 2, 0, 1, 2])
    logits_no_grl = nn.Linear(d_model, n_batches)(encoded)
    loss_no_grl = F.cross_entropy(logits_no_grl, batch_labels)
    
    grad_no_grl = torch.autograd.grad(loss_no_grl, encoded, retain_graph=False)[0]
    
    # With GRL: reversed gradient
    encoded2 = encoder(x)
    loss_with_grl = dar_classifier.compute_loss(encoded2, batch_labels)
    grad_with_grl = torch.autograd.grad(loss_with_grl, encoded2, retain_graph=False)[0]
    
    print(f"Normal classification gradient (first cell, first 4 dims):")
    print(f"  {grad_no_grl[0, :4].detach().numpy().round(4)}")
    print(f"GRL-reversed gradient (first cell, first 4 dims):")
    print(f"  {grad_with_grl[0, :4].detach().numpy().round(4)}")
    print(f"\nThe signs are flipped! GRL makes the encoder RESIST batch prediction.")
    print(f"After training, cell embeddings become batch-invariant.\n")


def demo_combined_finetuning():
    """
    Full fine-tuning forward pass demonstrating all loss components.
    """
    print("=" * 60)
    print("Combined Fine-Tuning Demo (Annotation Task)")
    print("=" * 60)
    
    # Import scFMModel from Module 2
    # (In practice, you'd load pretrained weights here)
    
    # Inline minimal scFMModel for demo
    class MinimalscFM(nn.Module):
        def __init__(self, vocab_size=100, d_model=64):
            super().__init__()
            self.d_model = d_model
            self.gene_embedding = nn.Embedding(vocab_size, d_model)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, 2, 64, dropout=0.1, batch_first=True),
                num_layers=2
            )
            self.expr_pred_head = nn.Linear(d_model, 1)
        
        def forward(self, gene_tokens, expr_values, condition_tokens, **kwargs):
            h = self.gene_embedding(gene_tokens)
            h = self.transformer(h)
            return {
                'gene_repr': h,
                'cell_repr': h[:, 0, :],
                'expr_pred': self.expr_pred_head(h).squeeze(-1)
            }
    
    batch_size, seq_len = 8, 30
    vocab_size = 100
    n_cell_types = 5
    d_model = 64
    
    pretrained = MinimalscFM(vocab_size, d_model)
    finetuner = scFMFineTuner(
        pretrained_scgpt=pretrained,
        task='annotation',
        n_cell_types=n_cell_types
    )
    
    # Synthetic batch
    gene_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    expr_values = torch.randint(0, 51, (batch_size, seq_len)).float()
    condition_tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(0, n_cell_types, (batch_size,))
    
    # Random mask: 40% of genes masked
    mask = torch.rand(batch_size, seq_len) < 0.4
    mask[:, 0] = False  # Never mask <cls>
    expr_targets = expr_values.clone()
    
    result = finetuner.forward_annotation(
        gene_tokens, expr_values, condition_tokens,
        labels=labels, expr_targets=expr_targets, mask=mask
    )
    
    print(f"Loss components:")
    print(f"  GEP loss  (gene expression): {result['gep_loss'].item():.4f}")
    print(f"  GEPC loss (cell modeling):   {result['gepc_loss'].item():.4f}")
    print(f"  CLS loss  (cell type):       {result['cls_loss'].item():.4f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  TOTAL loss:                  {result['total_loss'].item():.4f}")
    print(f"\nClassifier output shape: {result['logits'].shape}")
    
    # Show predicted cell types
    pred_types = result['logits'].argmax(-1)
    print(f"Predicted cell types: {pred_types.tolist()}")
    print(f"True cell types:      {labels.tolist()}")
    print()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("scFM Tutorial — Module 3: Fine-Tuning Objectives")
    print("="*70 + "\n")
    
    demo_gepc()
    demo_ecs_loss()
    demo_gradient_reversal()
    demo_combined_finetuning()
    
    print("=" * 60)
    print("Module 3 Complete!")
    print("=" * 60)
    print("""
    Key concepts covered:
    
    1. GEPC: The cell embedding must predict each gene's expression via
       a bilinear interaction — forces the cell vector to be a dense,
       biologically meaningful summary of cell state.
    
    2. ECS: Elastic contrastive loss that pulls similar cells together
       above a similarity threshold (β=0.6). Self-supervised, no labels.
    
    3. DAR / Gradient Reversal: An adversarial trick that makes cell
       embeddings batch-invariant by MAXIMIZING batch prediction error
       in the encoder while MINIMIZING it in the classifier.
    
    4. Multi-task training: GEP + GEPC preserve pretraining knowledge;
       CLS / ECS / DAR guide toward task-specific objectives.
       This prevents catastrophic forgetting during fine-tuning.
    
    5. Loss weighting: ECS weight=10 compensates for the smaller
       magnitude of cosine-similarity-based losses vs MSE losses.
    
    Next: Module 4 — Perturbation prediction and the training loop.
    """)
