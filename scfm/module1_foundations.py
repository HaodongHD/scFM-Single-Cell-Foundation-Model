"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         scFM — Single-Cell Foundation Model — Module 1: Foundations & Input Embeddings       ║
║         Based on: Cui et al., Nature Methods 2024                            ║
║         Code: https://github.com/bowang-lab/scGPT                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

LEARNING GOALS:
  1. Understand how scRNA-seq data is tokenized (gene tokens, expression values)
  2. Implement value-binning — scFM's key innovation for handling batch effects
  3. Build the three-part input embedding (gene + expression + condition)
  4. Understand WHY each design choice was made

PAPER SECTION COVERED: "Input Embeddings" (Methods section)

RUN THIS MODULE:
  pip install torch scanpy anndata numpy scipy
  python module1_foundations.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict
import math

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Understanding scRNA-seq data format
# ─────────────────────────────────────────────────────────────────────────────
"""
scRNA-seq data is a cell × gene matrix X ∈ R^(N×G)
  - N cells (rows)
  - G genes (columns)
  - Each value X[i,j] = RNA count for gene j in cell i

SPARSITY: Typically >90% zeros (most genes are not expressed in any given cell).
This is NOT like NLP where every word position has a token — many genes are simply
"off" in a cell. This is why scFM only processes *non-zero expressed* genes per cell
during pretraining (saves compute, focuses on signal).

KEY CHALLENGE 1: Batch effects
  Different sequencing runs produce different absolute count magnitudes.
  A count of "100" in batch A ≠ "100" in batch B biologically.
  Standard normalization (log1p, TPM) helps but doesn't fully solve this.

KEY CHALLENGE 2: Non-sequential nature
  Genes have no natural ordering (unlike words in a sentence).
  Can't use standard causal masking directly.
  scFM's solution: generative attention masking (Module 2).
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Gene Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

class GeneVocab:
    """
    Maps gene names ↔ integer token IDs.

    WHY: Transformer models operate on integer indices into an embedding table.
    Each gene (like 'TP53', 'GAPDH') gets a unique integer ID.
    This is exactly like a word vocabulary in NLP.

    SCALE: Human genome ~20,000 protein-coding genes.
    scFM uses the full human genome as vocabulary (unlike Geneformer which
    ranks by expression — scFM keeps raw gene identity).

    SPECIAL TOKENS:
      <cls>  — aggregates cell-level representation (like BERT's [CLS])
      <pad>  — pads sequences to fixed length
      <mask> — used during masked gene prediction fine-tuning
    """

    def __init__(self, gene_list: List[str]):
        # Reserve IDs 0,1,2 for special tokens
        self.special_tokens = {"<pad>": 0, "<cls>": 1, "<mask>": 2}
        self.gene2id = {**self.special_tokens}
        self.id2gene = {v: k for k, v in self.special_tokens.items()}

        for idx, gene in enumerate(gene_list, start=len(self.special_tokens)):
            self.gene2id[gene] = idx
            self.id2gene[idx] = gene

    def __len__(self):
        return len(self.gene2id)

    def __getitem__(self, gene: str) -> int:
        # Returns <pad> ID (0) for unknown genes — safe fallback
        return self.gene2id.get(gene, self.special_tokens["<pad>"])

    def tokenize(self, expressed_genes: List[str]) -> torch.Tensor:
        """Convert list of gene names to token IDs."""
        # Prepend <cls> token — this position will become the cell embedding
        tokens = [self.special_tokens["<cls>"]] + [self[g] for g in expressed_genes]
        return torch.tensor(tokens, dtype=torch.long)


# Demonstration
print("=" * 70)
print("SECTION 2: Gene Vocabulary")
print("=" * 70)
example_genes = ["TP53", "GAPDH", "CD3E", "CD8A", "IL6", "TNF", "VEGFA"]
vocab = GeneVocab(example_genes)
print(f"Vocabulary size: {len(vocab)} (3 special + {len(example_genes)} genes)")
print(f"TP53 token ID: {vocab['TP53']}")
print(f"Token IDs for ['CD3E', 'CD8A']: {vocab.tokenize(['CD3E', 'CD8A'])}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Value Binning — scFM's Core Contribution for Batch Robustness
# ─────────────────────────────────────────────────────────────────────────────

def value_binning(expression_values: np.ndarray, n_bins: int = 51) -> np.ndarray:
    """
    Converts raw expression counts → relative bin indices.

    WHY THIS MATTERS (from paper):
      "Even after [log1p transformation], the same absolute value can convey
       different 'semantic' meanings across sequencing batches."

    SOLUTION: Instead of absolute counts, use RELATIVE ranks within each cell.
      - Divide expressed genes into B equal-frequency bins per cell
      - Bin B = "highest expressed" genes, Bin 1 = "lowest expressed"
      - Zero expression → bin 0 (special "not expressed" token)

    This is similar in spirit to Geneformer's rank-encoding, but:
      - Geneformer: ranks 1..N_expressed (integer rank)
      - scFM: bins the ranks into B groups (B=51 typically)
      - Binning is more robust to outliers and gives ordinal embedding semantics

    ANALOGY: Like converting test scores to letter grades (A/B/C/D/F).
    "90% in test A" and "90% in test B" are comparable grades even if the
    tests have different difficulty. Same idea here across batches.

    Args:
        expression_values: 1D array of expression counts for ONE cell
                           (only non-zero values should be passed)
        n_bins: Number of bins. Paper uses 51 (bin 0 reserved for zero expression)

    Returns:
        binned: Same shape array with values in [0, n_bins-1]
                0 = not expressed, 1..n_bins-1 = expression bins
    """
    binned = np.zeros_like(expression_values, dtype=np.int64)
    nonzero_mask = expression_values > 0
    nonzero_vals = expression_values[nonzero_mask]

    if len(nonzero_vals) == 0:
        return binned

    # IMPORTANT: bin edges are computed PER CELL from the distribution of
    # expressed genes. This makes the encoding relative within each cell.
    # Use percentile-based bin edges (equal frequency binning)
    n_nonzero = len(nonzero_vals)
    # n_bins - 1 actual bins (bin 0 is reserved for zeros)
    n_actual_bins = n_bins - 1

    # Create equal-frequency bin edges using percentiles
    # linspace(0, 100, n+1) gives n+1 percentile points → n intervals
    percentiles = np.linspace(0, 100, n_actual_bins + 1)
    bin_edges = np.percentile(nonzero_vals, percentiles)

    # Assign each nonzero value to a bin (1-indexed so 0 means "not expressed")
    # np.digitize returns 1..n_bins for values in range
    bin_assignments = np.digitize(nonzero_vals, bin_edges[1:-1]) + 1
    # Clip to valid range [1, n_bins-1]
    bin_assignments = np.clip(bin_assignments, 1, n_bins - 1)
    binned[nonzero_mask] = bin_assignments

    return binned


# Demonstration
print("\n" + "=" * 70)
print("SECTION 3: Value Binning")
print("=" * 70)

# Simulate two cells from different batches with different absolute counts
# But biologically similar (same relative expression pattern)
np.random.seed(42)
# Cell from batch A (low depth): counts 1-100
cell_A = np.array([0, 0, 5, 10, 50, 100, 0, 2, 20, 80], dtype=float)
# Cell from batch B (high depth): same relative pattern, 10x higher counts
cell_B = np.array([0, 0, 50, 100, 500, 1000, 0, 20, 200, 800], dtype=float)

binned_A = value_binning(cell_A, n_bins=6)
binned_B = value_binning(cell_B, n_bins=6)

print(f"Cell A (low depth):  {cell_A.astype(int)}")
print(f"Cell B (high depth): {cell_B.astype(int)}")
print(f"Binned A: {binned_A}")
print(f"Binned B: {binned_B}")
print("→ Same bin assignments despite 10x different absolute counts!")
print("→ This is WHY binning helps with batch effects.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: The Three-Part Input Embedding
# ─────────────────────────────────────────────────────────────────────────────

class scFMInputEmbedding(nn.Module):
    """
    Combines three sources of information into a single embedding per gene position.

    From paper (Eq. 5):
        h_i = emb_g(t_g^i) + emb_x(x^i) + emb_c(t_c^i)

    WHERE:
        emb_g  — Gene token embedding (nn.Embedding): WHAT gene is at this position?
        emb_x  — Expression value embedding (nn.Linear): HOW expressed is it?
        emb_c  — Condition token embedding (nn.Embedding): WHAT context? (batch/perturbation)

    DESIGN DECISIONS:
    ┌──────────────────────────────────────────────────────────────────┐
    │ emb_g: nn.Embedding (discrete lookup)                            │
    │   → Gene identity is categorical, not ordinal. Each gene has    │
    │   → a unique learned representation. Like word embeddings in NLP│
    │   → Dimension D=512 in paper                                    │
    │                                                                  │
    │ emb_x: nn.Linear (fully connected layer)                        │
    │   → Expression level IS ordinal (bin 5 > bin 3 biologically)   │
    │   → nn.Linear better captures this ordinal relationship than    │
    │   → a discrete lookup table (which would treat each bin as      │
    │   → independent).                                                │
    │   → Input: scalar bin index → Output: D-dim vector             │
    │                                                                  │
    │ emb_c: nn.Embedding (discrete lookup)                           │
    │   → Condition (batch, perturbation) is categorical              │
    │   → Added elementwise to allow conditioning without changing    │
    │   → the transformer architecture                                 │
    └──────────────────────────────────────────────────────────────────┘

    ALTERNATIVE CONSIDERED: Concatenation instead of addition.
    Addition saves memory (no dimension increase) and allows each embedding
    to specialize while interacting through the transformer attention.
    This is the same design as BERT (token + position + segment embeddings summed).

    Args:
        vocab_size: Size of gene vocabulary (|genes| + special tokens)
        d_model: Embedding dimension. Paper uses 512.
                 TUNING TIP: 256 for small models, 512 paper, 1024 for larger.
                 Must be divisible by n_heads in transformer.
        n_bins: Number of expression bins (paper: 51, including bin 0 for zeros)
        n_conditions: Number of condition categories (batches, perturbations, modalities)
        padding_idx: Token ID for <pad> — gets zero embedding, no gradient
        dropout: Applied after summing embeddings. Regularization.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_bins: int = 51,
        n_conditions: int = 10,
        padding_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # ── Gene token embedding: gene_id → d_model vector ────────────────
        # padding_idx=0 ensures <pad> always has zero embedding
        self.gene_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )

        # ── Expression value embedding: bin_index → d_model vector ────────
        # Using Linear instead of Embedding to capture ordinal relationships.
        # Input is a single float (the bin index), output is d_model dim vector.
        # WHY NOT Embedding here? Embedding(51) would work too, but Linear
        # can generalize across unseen values and better captures monotonicity.
        self.expr_embedding = nn.Linear(1, d_model, bias=True)

        # ── Condition token embedding: condition_id → d_model vector ──────
        # Conditions include: batch ID, modality (RNA/ATAC/protein),
        # perturbation status (0=control, 1=perturbed gene)
        self.condition_embedding = nn.Embedding(n_conditions, d_model)

        # ── Layer Normalization + Dropout ─────────────────────────────────
        # LayerNorm stabilizes training when summing three different embeddings
        # that may have different scales. Applied AFTER summing.
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # ── Weight initialization ─────────────────────────────────────────
        # Important: initialize embeddings with small values to prevent
        # any single embedding from dominating at the start of training
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.gene_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.expr_embedding.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.expr_embedding.bias, 0.0)
        nn.init.normal_(self.condition_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        gene_tokens: torch.Tensor,    # (batch, seq_len) — integer gene IDs
        expr_values: torch.Tensor,    # (batch, seq_len) — integer bin indices
        condition_tokens: torch.Tensor,  # (batch, seq_len) — integer condition IDs
    ) -> torch.Tensor:
        """
        Returns: embedding tensor of shape (batch, seq_len, d_model)

        Each position gets: gene_emb + expr_emb + condition_emb
        """
        B, L = gene_tokens.shape

        # Gene embedding: (B, L) → (B, L, d_model)
        g_emb = self.gene_embedding(gene_tokens)

        # Expression embedding: bin index → float → linear → (B, L, d_model)
        # Must convert int bins to float and add feature dimension
        x_float = expr_values.float().unsqueeze(-1)  # (B, L, 1)
        x_emb = self.expr_embedding(x_float)          # (B, L, d_model)

        # Condition embedding: (B, L) → (B, L, d_model)
        c_emb = self.condition_embedding(condition_tokens)

        # Element-wise sum (Eq. 5 from paper)
        h = g_emb + x_emb + c_emb    # (B, L, d_model)

        # Normalize and regularize
        h = self.layer_norm(h)
        h = self.dropout(h)

        return h


# Demonstration
print("\n" + "=" * 70)
print("SECTION 4: Input Embedding")
print("=" * 70)

VOCAB_SIZE = 20000  # Human genome
D_MODEL = 512
N_BINS = 51
BATCH_SIZE = 4
SEQ_LEN = 200  # Max expressed genes per cell

embedding_layer = scFMInputEmbedding(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_bins=N_BINS,
    n_conditions=10,
)

# Simulate a batch of 4 cells, each with 200 expressed genes
gene_tokens = torch.randint(3, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
gene_tokens[:, 0] = 1  # First position = <cls> token for all cells

expr_values = torch.randint(0, N_BINS, (BATCH_SIZE, SEQ_LEN))
expr_values[:, 0] = 0  # <cls> position has no expression value

condition_tokens = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)

h = embedding_layer(gene_tokens, expr_values, condition_tokens)
print(f"Input:  gene_tokens {gene_tokens.shape}, expr_values {expr_values.shape}")
print(f"Output: embedding {h.shape}  (batch=4, seq_len=200, d_model=512)")
print(f"Memory: {h.element_size() * h.nelement() / 1024:.1f} KB per batch")
print(f"\nParameter count in embedding layer:")
total_params = sum(p.numel() for p in embedding_layer.parameters())
print(f"  gene_embedding:      {embedding_layer.gene_embedding.weight.numel():>10,}")
print(f"  expr_embedding:      {sum(p.numel() for p in embedding_layer.expr_embedding.parameters()):>10,}")
print(f"  condition_embedding: {embedding_layer.condition_embedding.weight.numel():>10,}")
print(f"  TOTAL:               {total_params:>10,}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Why NOT positional encoding?
# ─────────────────────────────────────────────────────────────────────────────
"""
CRITICAL DIFFERENCE from standard NLP transformers:

Standard Transformer (BERT/GPT): adds sinusoidal or learned POSITIONAL embeddings
  h_i = token_emb(w_i) + pos_emb(i)

scFM: NO positional encoding!

WHY? Gene expression is NON-SEQUENTIAL.
  - "TP53 is at position 5" is meaningless — genes don't have positions
  - The ORDER of gene tokens in the input is arbitrary
  - What matters is the SET of expressed genes and their relative levels
  - Adding positional encoding would introduce spurious positional dependencies

This is a key architectural decision. Without positional encoding, the
transformer must rely entirely on gene identity (which gene) and attention
between genes to learn relationships — exactly what we want for GRN inference!

PRACTICAL IMPLICATION: You can shuffle the input gene order and get the
same output (up to permutation of output positions). This is set-equivariant
processing, which is the correct inductive bias for gene expression data.

The attention mask (Module 2) handles the generative ordering, not positional
encoding.
"""
print("\n" + "=" * 70)
print("SECTION 5: Set-equivariance demonstration (no positional encoding)")
print("=" * 70)
# Demonstrate set-equivariance: shuffling input genes should give same embeddings
gene_tokens_shuffled = gene_tokens[:, 1:].clone()  # exclude <cls>
perm = torch.randperm(SEQ_LEN - 1) + 1  # don't shuffle <cls>
gene_tokens_perm = gene_tokens.clone()
gene_tokens_perm[:, 1:] = gene_tokens_shuffled[:, perm - 1]
expr_values_perm = expr_values.clone()
expr_values_perm[:, 1:] = expr_values[:, 1:][:, perm - 1]
cond_perm = condition_tokens.clone()

h_perm = embedding_layer(gene_tokens_perm, expr_values_perm, cond_perm)
print("Embedding of gene at position 1 (original):", h[0, 1, :4].detach().numpy().round(3))
print("Embedding of same gene shuffled to position 5:", h_perm[0, 5, :4].detach().numpy().round(3))
print("→ Same embedding regardless of position (no positional encoding)!")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Putting it all together — data preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────────────

class scFMDataPreprocessor:
    """
    Full preprocessing pipeline: raw AnnData → model-ready tensors.

    This mirrors the scFM preprocessing in their actual codebase.
    Understanding this pipeline is crucial for applying scFM to your own data.

    STEPS:
    1. Filter low-quality cells (min genes expressed)
    2. Normalize per cell (total count normalization)
    3. Log1p transform  (reduces dynamic range, makes distribution more normal)
    4. Select highly variable genes (HVGs) — typically 1200-3000
    5. Value binning (scFM's key step)
    6. Tokenize: gene names → integer IDs
    7. Pad/truncate to max_length
    """

    def __init__(
        self,
        vocab: GeneVocab,
        n_bins: int = 51,
        max_seq_len: int = 1200,  # Paper uses 1200 — covers most expressed gene sets
        n_hvg: int = 1200,        # Number of highly variable genes to select
    ):
        self.vocab = vocab
        self.n_bins = n_bins
        self.max_seq_len = max_seq_len  # includes <cls> token
        self.n_hvg = n_hvg

    def process_cell(
        self,
        gene_names: List[str],
        expression_counts: np.ndarray,
        condition_id: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single cell's expression data into model inputs.

        Args:
            gene_names: List of gene names (same order as expression_counts)
            expression_counts: Raw/normalized expression values (1D array)
            condition_id: Integer indicating batch/condition (for condition token)

        Returns dict with:
            gene_tokens:      (max_seq_len,) int64
            expr_values:      (max_seq_len,) int64
            condition_tokens: (max_seq_len,) int64
            padding_mask:     (max_seq_len,) bool — True = padded (should be ignored)
        """
        # Step 1: Get expressed genes (non-zero)
        expressed_mask = expression_counts > 0
        expressed_genes = [g for g, m in zip(gene_names, expressed_mask) if m]
        expressed_vals = expression_counts[expressed_mask]

        # Step 2: Value binning
        binned_vals = value_binning(expressed_vals, n_bins=self.n_bins)

        # Step 3: Tokenize genes → IDs
        gene_ids = [self.vocab[g] for g in expressed_genes]

        # Step 4: Truncate to max_seq_len - 1 (leave room for <cls>)
        max_genes = self.max_seq_len - 1
        if len(gene_ids) > max_genes:
            # PAPER STRATEGY: randomly sample when too many expressed genes
            # This is done fresh at each training iteration (data augmentation!)
            indices = np.random.choice(len(gene_ids), max_genes, replace=False)
            indices.sort()  # Keep genes in consistent order within a cell
            gene_ids = [gene_ids[i] for i in indices]
            binned_vals = binned_vals[indices]

        actual_len = len(gene_ids) + 1  # +1 for <cls>

        # Step 5: Build tensors with padding
        gene_tensor = torch.zeros(self.max_seq_len, dtype=torch.long)
        expr_tensor = torch.zeros(self.max_seq_len, dtype=torch.long)
        cond_tensor = torch.full((self.max_seq_len,), condition_id, dtype=torch.long)
        pad_mask = torch.ones(self.max_seq_len, dtype=torch.bool)

        # Position 0: <cls> token
        gene_tensor[0] = self.vocab.special_tokens["<cls>"]
        expr_tensor[0] = 0  # <cls> has no expression level
        pad_mask[0] = False  # <cls> is NOT padding

        # Positions 1..actual_len: expressed genes
        gene_tensor[1:actual_len] = torch.tensor(gene_ids, dtype=torch.long)
        expr_tensor[1:actual_len] = torch.tensor(binned_vals, dtype=torch.long)
        pad_mask[1:actual_len] = False  # These positions are real data

        # Positions actual_len..max_seq_len: padding
        gene_tensor[actual_len:] = self.vocab.special_tokens["<pad>"]
        # pad_mask[actual_len:] = True already set above

        return {
            "gene_tokens": gene_tensor,
            "expr_values": expr_tensor,
            "condition_tokens": cond_tensor,
            "padding_mask": pad_mask,
            "actual_length": actual_len,
        }


# Demonstration
print("\n" + "=" * 70)
print("SECTION 6: Full preprocessing pipeline")
print("=" * 70)

# Simulate one cell
np.random.seed(0)
n_genes = 500
gene_names_example = [f"GENE_{i}" for i in range(n_genes)]
raw_counts = np.random.negative_binomial(1, 0.3, n_genes).astype(float)
raw_counts[raw_counts == 0] = 0  # Keep sparsity

vocab_example = GeneVocab(gene_names_example)
preprocessor = scFMDataPreprocessor(vocab=vocab_example, n_bins=51, max_seq_len=256)
processed = preprocessor.process_cell(gene_names_example, raw_counts, condition_id=0)

n_expressed = (raw_counts > 0).sum()
n_actual = processed["actual_length"]
print(f"Raw expression: {n_genes} genes, {n_expressed} expressed")
print(f"After processing:")
print(f"  gene_tokens shape:  {processed['gene_tokens'].shape}")
print(f"  expr_values shape:  {processed['expr_values'].shape}")
print(f"  Actual positions:   {processed['actual_length']} (including <cls>)")
print(f"  Padded positions:   {processed['padding_mask'].sum().item()}")
print(f"  First 5 bins: {processed['expr_values'][1:6].tolist()}")


print("\n✅ Module 1 complete! Key takeaways:")
print("  1. Gene names → integer IDs (GeneVocab)")
print("  2. Value binning makes expression RELATIVE within each cell")
print("     → Solves batch effect problem in input representation")
print("  3. Three-part embedding: gene identity + expression level + condition")
print("  4. NO positional encoding — genes are a set, not a sequence")
print("  5. Padding mask tracks real vs padded positions")
print("\nNext: Module 2 — The Masked Attention Transformer")
