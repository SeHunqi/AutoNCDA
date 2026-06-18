
## Repository Structure

```
AutoNCDA/
├── data/
│   ├── dataset.py              # Base RNA dataset (family task, pre-extracted .npy)
│   ├── dataset_lookup.py       # Lookup dataset for association task (dynamic negatives)
│   └── dataloader_factory.py  # 5-fold cross-validation data loader factory
├── models/
│   ├── model.py                # Core MMoE model (MMoEModel + StaticMMoEModel)
│   └── nas/
│       ├── mix_expert.py       # NAS operation pool (SwiGLU, ResNeXt, GatedMLP, ...)
│       ├── mix_share.py        # Searchable sharing pattern module
│       └── mix_adapter.py      # Searchable bottleneck adapter
├── training/
│   ├── losses.py               # FocalLoss, Mixup, temperature scaling, threshold search
│   ├── metrics.py              # Metric logging utilities
│   ├── trainer.py              # DecoupledTrainer (main training loop)
│   └── search_trainer.py       # DARTS-like bilevel NAS searcher
├── scripts/
│   ├── train.py                # Main training entry point
│   └── generate_dss.py         # DSS feature generation (Semantic + GIP similarity)
└── utils/
    └── arch_utils.py           # NAS architecture sampling and export utilities
```




