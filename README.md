# scFM — Single-Cell Foundation Model

A from-scratch implementation of a **generative foundation model for single-cell RNA sequencing (scRNA-seq)**, built in PyTorch with step-by-step commentary explaining every architectural decision.

---

## The implementation has 4 modules:

| Module | Topic | Key Concepts |
|--------|-------|--------------|
| **Module 1** | Foundations & Input Embeddings | Gene vocabulary, value binning, three-part embedding |
| **Module 2** | The Masked Attention Transformer | Generative attention mask, multi-head attention, GEP pretraining |
| **Module 3** | Fine-Tuning Objectives | GEPC, ECS (elastic contrastive), gradient reversal (DAR), classification |
| **Module 4** | Training, Perturbation & GRN | AdamW loop, Pearson_delta, attention-based GRN inference |

---

## Architecture Overview

```
scRNA-seq Cell  →  Tokenization  →  Three-Part Embedding  →  Transformer (12L × 8H)  →  Task Heads
                                                                                          ├── GEP  (expression prediction)
                                                                                          ├── GEPC (cell-level prediction)
                                                                                          ├── ECS  (batch integration)
                                                                                          ├── DAR  (domain adaptation)
                                                                                          └── CLS  (cell type annotation)
```

**Key architectural choices implemented and explained:**
- **No positional encoding** — genes are a *set*, not a sequence
- **Value binning** — converts raw counts to relative expression ranks, removing batch effects at the input level
- **Generative attention mask** — enables pretraining on non-sequential gene sets
- **`<cls>` token** — aggregates all gene context into a single cell-level representation


---

## Module Summaries

### Module 1 — Foundations & Input Embeddings

- `GeneVocab` — maps gene names to integer IDs, with special tokens (`<pad>`, `<cls>`, `<mask>`)
- `value_binning` — converts absolute expression counts to **relative bin indices per cell**, the core trick for batch robustness:

  ```python
  # Same biology, two different sequencing depths:
  cell_A = np.array([0, 5, 10, 50, 100])     # batch A
  cell_B = np.array([0, 50, 100, 500, 1000])  # batch B (10x higher)

  binned_A = value_binning(cell_A, n_bins=6)  # [0, 1, 2, 4, 5]
  binned_B = value_binning(cell_B, n_bins=6)  # [0, 1, 2, 4, 5]  ← identical
  ```

- `scFMInputEmbedding` — combines three sources into a single vector per gene position:

  ```
  h_i = emb_g(gene_id)  +  emb_x(bin_value)  +  emb_c(condition_id)
          ↑ categorical       ↑ ordinal (Linear)    ↑ categorical
  ```

- `scFMDataPreprocessor` — end-to-end pipeline from raw counts to model-ready tensors

---

### Module 2 — The Masked Attention Transformer

- **Why not causal masking?** — genes have no natural order; positional masking would introduce spurious dependencies
- **Generative attention mask** — tokens are split into *known* (prompt) and *unknown* (target):

  ```
  Mask (K=known, U=unknown):
              <cls>  G1(K)  G2(K)  G3(U)  G4(U)
  G3(U)       attend attend attend attend  BLOCK
  G4(U)       attend attend attend  BLOCK attend
  ```

- `scFMTransformerBlock` — Pre-LayerNorm for stability in 12-layer networks
- `GeneExpressionPredictionHead` — MLP with **MSE loss** (not cross-entropy) to preserve the ordinal structure of expression bins
- Complete training step with gradient clipping (`max_norm=1.0`)

---

### Module 3 — Fine-Tuning Objectives

Four composable objectives for different downstream tasks:

| Loss | Type | Purpose |
|------|------|---------|
| GEP  | Self-supervised | Predict masked expression from gene-level context |
| GEPC | Self-supervised | Predict expression from the **cell embedding** — forces rich cell representations |
| ECS  | Contrastive | Pull biologically similar cells together above similarity threshold β |
| DAR  | Adversarial | Remove batch effects via gradient reversal |

**ECS — Elastic Cell Similarity:**
```
L_ECS = -mean( (cosine_sim(h_c_i, h_c_j) - β)² )
```

**DAR — gradient reversal for batch correction:**
```python
x_reversed = GradientReversalLayer()(cell_embedding)
# Encoder learns to make batch prediction HARDER → embeddings become batch-invariant
```

**Task combinations:**
- Cell type annotation: `GEP + GEPC + CLS`
- Batch integration:    `GEP + GEPC + 10×ECS + DAR`

---

### Module 4 — Training, Perturbation & GRN

- `scFMTrainingConfig` — all hyperparameters with justifications
- `scFMTrainer` — full loop: AdamW, step LR decay (×0.9/epoch), gradient clipping, checkpointing
- **Variable mask ratios** — uniformly sampled from `{0.25, 0.50, 0.75}` per step
- **Perturbation prediction** — control expression + knockout token → predicted post-perturbation profile; evaluated with `Pearson_delta`
- **GRN inference** — rank-normalized attention maps identify transcription factor regulatory targets

---


## Repository Structure

```
scfm/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── .gitignore
└── scfm/
    ├── __init__.py
    ├── module1_foundations.py      # Vocabulary, binning, embeddings
    ├── module2_transformer.py      # Masked attention, GEP loss
    ├── module3_finetuning.py       # GEPC, ECS, DAR, classification
    └── module4_training_grn.py     # Training loop, perturbation, GRN
```



## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy, SciPy, scikit-learn
- Scanpy + AnnData

See `requirements.txt` for full list.

---

## License

MIT License. See `LICENSE` for details.
