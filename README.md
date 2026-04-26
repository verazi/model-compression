# Model Compression: ResNet-50 → MobileNetV3-Small

Knowledge Distillation + Pruning + Quantization on CIFAR-100  
Targeting edge deployment on resource-constrained devices

---

## Results

### Compression Pipeline

| Stage | Accuracy | Size | Latency | Size↓ | Acc Drop |
|-------|----------|------|---------|-------|----------|
| Teacher (ResNet-50) | 75.22% | 90.6 MB | 7.1 ms | 1x | — |
| Student Fine-tune (baseline) | ~68% | 6.2 MB | 3.3 ms | 14.6x | ~7% |
| Student KD (scratch) | 45.49% | 6.2 MB | 3.3 ms | 14.6x | 29.7% |
| Student KD (pretrained) | 69.53% | 6.2 MB | 3.3 ms | 14.6x | 5.7% |
| + Pruning (30%) | 67.89% | 6.2 MB | 3.3 ms | 14.6x | 7.3% |
| + FP16 (ONNX) | 67.91% | 3.1 MB | 3.0 ms | 29x | 7.3% |
| + INT8 PTQ (ONNX) | 62.97% | 1.9 MB | 3.3 ms | 48x | 12.3% |

![Visualization](https://github.com/verazi/model-compression/blob/main/results/compression_pipeline_visualization.png)

### Quantization Comparison

| Method | Accuracy | Size | Latency | Size↓ | Notes |
|--------|----------|------|---------|-------|-------|
| FP32 (baseline) | 67.89% | 6.19 MB | 3.33 ms | 1x | — |
| FP16 | 67.91% | 3.13 MB | 2.98 ms | 2.0x | Near-zero accuracy loss |
| INT8 PTQ | 62.97% | 1.89 MB | 3.25 ms | 3.3x | MobileNet quantization-sensitive |

> **Key Finding**: MobileNetV3's Depthwise Separable Conv and SE Module are sensitive to INT8 PTQ.  
> FP16 is the recommended choice for accuracy-critical edge deployment.  
> INT8 requires QAT for acceptable accuracy — listed as future work.

### KD Ablation Study

| Init | Accuracy | Epochs | Training Time |
|------|----------|--------|---------------|
| weights=None (scratch) | 45.49% | 50 | ~75 min |
| ImageNet pretrained | 69.53% | 80 | ~76 min |

> ImageNet pretrained initialization provides **+24% accuracy** with the same training time,  
> as the student starts with strong visual priors instead of learning from scratch.

---

## Architecture

```
CIFAR-100 Dataset (50K train / 10K val, 100 classes, 32×32)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Phase 1: Model Compression                         │
│                                                     │
│  ResNet-50 (ImageNet pretrained)                    │
│    → Transfer Learning (freeze→unfreeze, 30 epochs) │
│    → Teacher: 75.22% on CIFAR-100                   │
│         │                                           │
│         │ Knowledge Distillation (soft labels, T=4) │
│         ▼                                           │
│  MobileNetV3-Small (ImageNet pretrained)            │
│    → KD Training (80 epochs)                        │
│    → Student: 69.53%                                │
│         │                                           │
│         │ L1 Unstructured Pruning (30%)             │
│         ▼                                           │
│  Pruned Student: 67.89%, params 1.62M               │
│         │                                           │
│    ┌────┴────┐                                      │
│    ▼         ▼                                      │
│  FP16      INT8 PTQ                                 │
│  67.91%    62.97%                                   │
│  3.1MB     1.9MB                                    │
└─────────────────────────────────────────────────────┘
```

---

## Pipeline Design Rationale

### Why Knowledge Distillation?

Direct fine-tuning MobileNetV3 on CIFAR-100 achieves ~67-69%. KD adds ~1-2% by having the student learn from the teacher's **output distribution** (soft labels), not just hard one-hot labels.

Soft labels encode inter-class similarity:
```
Hard label:  [0, 0, 0, 1, 0, ...]       # only "cat"
Soft label:  [0.01, 0.42, 0.33, ...]    # cat 42%, leopard 33%, ...
```

This richer training signal helps the small model generalize better with the same number of parameters.

### Why Pruning?

L1 Unstructured Pruning removes weights with lowest L1 norm (30%), reducing parameter count from 2.5M → 1.62M (-35%).

**Important limitation**: Unstructured pruning does not reduce latency on standard hardware (GPU/CPU cannot exploit sparse matrices). Structured Pruning (removing entire filters) would give real latency improvements — listed as future work.

Pruning also has a regularization effect: the train/val accuracy gap narrows slightly after pruning, as removing low-magnitude weights eliminates some overfitting noise.

### Why FP16 over INT8?

| | FP16 | INT8 PTQ |
|--|------|----------|
| Accuracy drop | ~0% | ~5% |
| Size reduction | 2x | 3.5x |
| Latency on CPU | similar | similar |
| Calibration needed | No | Yes |

For MobileNetV3 specifically, INT8 PTQ causes ~5% accuracy degradation due to:
1. **Depthwise Conv**: only 9 weights per filter → quantization error cannot average out
2. **SE Module**: Sigmoid output in [0,1] → only 128 discrete values in INT8
3. **h-swish**: similar [0,1] range issue

**FP16 is the pragmatic choice** for this architecture. INT8 requires QAT to be effective.

---

## Experiment Tracking

All experiments tracked with MLflow:
- Hyperparameters: lr, epochs, temperature, alpha, prune_amount
- Metrics: train_loss, train_acc, val_loss, val_acc, epoch_time

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Setup

### Requirements
```bash
pip install torch torchvision torchinfo thop mlflow
pip install onnx onnxruntime onnxscript onnxconverter-common
```

### Run on Google Colab
1. Open `project_final.py` in Colab
2. Mount Google Drive (checkpoints saved automatically)
3. Run all cells in order — each cell checks for existing checkpoints and skips training if found

### Checkpoint structure
```
ml_pipeline/
├── checkpoints/
│   ├── teacher_best.pth                    # ResNet-50 Teacher
│   ├── student_kd_scratch_best.pth         # KD without pretrained (ablation)
│   ├── student_kd_pretrained_best.pth      # KD with pretrained (main)
│   ├── student_pruned_best.pth             # After Pruning
│   ├── student_pruned_fp32.onnx            # FP32 deployment
│   ├── student_pruned_fp16.onnx            # FP16 deployment
│   └── student_pruned_int8.onnx            # INT8 deployment
├── results/
│   ├── benchmark_results.csv
│   ├── quantization_comparison.csv
│   └── compression_pipeline_visualization.png
└── mlflow.db
```

---

## Key Observations

1. **KD vs Direct Fine-tune**: KD with pretrained initialization outperforms direct fine-tuning by ~1-2%, validating that soft labels transfer inter-class knowledge effectively.

2. **Pretrained initialization matters more than KD**: The gap between scratch (45.49%) and pretrained (69.53%) KD is 24%, showing that visual priors from ImageNet dominate over training strategy for small models.

3. **MobileNetV3 is quantization-unfriendly**: INT8 PTQ causes 5% accuracy drop vs <0.1% for FP16. This is a known issue with Depthwise Separable Conv architectures and motivates QAT as the correct quantization approach.

4. **Pruning regularization effect**: Train/val accuracy gap narrows from 21% → ~25% difference after pruning — unexpected result suggesting the removed weights were partly overfitting noise. Needs further investigation.

---

## Future Work

- [ ] **Structured Pruning** (torch-pruning): Remove entire filters for real latency reduction
- [ ] **QAT** (Quantization-Aware Training): Target INT8 accuracy drop < 2%
- [ ] **Mixup / CutMix**: Reduce overfitting (train-val gap ~21%)
- [ ] **Mixed Precision Training**: Speed up training 1.5-2x
- [ ] **Triton Inference Server**: Deploy FP16 ONNX, measure throughput/latency
- [ ] **DDP Multi-GPU Training**: Use NCCL AllReduce (connects to MPI background)
- [ ] **Prometheus + Grafana**: Monitor inference latency and GPU utilization
ers, memory access is the bottleneck) directly informs why FP16 outperforms INT8 PTQ for MobileNet: the issue is not compute, but the precision of weight representation in small convolutional filters.
