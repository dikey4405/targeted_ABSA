# Graph Report - targeted_ABSA  (2026-06-27)

## Corpus Check
- 15 files · ~8,193 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 142 nodes · 215 edges · 11 communities
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ca8630bb`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]

## God Nodes (most connected - your core abstractions)
1. `Vocabulary` - 25 edges
2. `TargetedABSAModel` - 14 edges
3. `ABSATrainer` - 14 edges
4. `Architecture` - 10 edges
5. `Core Components` - 10 edges
6. `ABSATargetedDataset` - 9 edges
7. `Technology Stack` - 8 edges
8. `main()` - 7 edges
9. `AttentionPooling` - 6 edges
10. `ConditionalAttention` - 6 edges

## Surprising Connections (you probably didn't know these)
- `ABSATrainer` --uses--> `TargetedABSAModel`  [INFERRED]
  train.py → model/targeted_absa.py
- `FocalLoss` --uses--> `TargetedABSAModel`  [INFERRED]
  train.py → model/targeted_absa.py
- `FocalLoss` --uses--> `Vocabulary`  [INFERRED]
  train.py → vocabulary.py
- `ABSATrainer` --uses--> `Vocabulary`  [INFERRED]
  train.py → vocabulary.py
- `main()` --calls--> `build_dataloader()`  [EXTRACTED]
  train.py → dataloader.py

## Import Cycles
- None detected.

## Communities (11 total, 0 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.17
Nodes (7): Any, Trả về annotation đã tách rõ target, aspect, sentiment để dùng cho ABSA targeted, Chuyển đổi danh sách các chuỗi nhãn thành một Tensor nhị phân (Multi-hot vector), Đọc dữ liệu từ file JSON Lines; vẫn hỗ trợ JSON list/dict nếu cần., Tách nhãn đầy đủ thành aspect và sentiment.         Ví dụ: ROOMS#CLEANLINESS#POS, Trích xuất danh sách các chuỗi nhãn đầy đủ từ trường 'label' của một sample., Vocabulary

### Community 1 - "Community 1"
Cohesion: 0.15
Nodes (5): AttentionPooling, ConditionalAttention, GatedFusion, MLPHead, TargetedABSAModel

### Community 2 - "Community 2"
Cohesion: 0.17
Nodes (7): DataLoader, device, Module, ABSATrainer, FocalLoss, main(), Hàm Loss linh hoạt dựa trên loss_key từ config

### Community 3 - "Community 3"
Cohesion: 0.13
Nodes (14): Anti-Patterns, Architectural Constraints, Architecture, Cross-Cutting Concerns, Data Flow, Design Pattern, Error Handling, Evaluation / Inference Pipeline (+6 more)

### Community 4 - "Community 4"
Cohesion: 0.23
Nodes (7): absa_collate_fn(), ABSATargetedDataset, build_dataloader(), Dataset cho targeted ABSA.      Mỗi annotation [start, end, label] trong một rev, Collate batch gồm tensor và metadata string.     Tensor được stack; các field te, Dataset, Path

### Community 5 - "Community 5"
Cohesion: 0.17
Nodes (11): `config/` — Experiment Configuration Layer, `dataloader.py` — Batch Construction Layer, Directory Layout, Key Files, `model/` — Model Architecture Layer, Module Boundaries, Naming Conventions, Project Structure (+3 more)

### Community 6 - "Community 6"
Cohesion: 0.20
Nodes (10): 1. `Vocabulary` — `vocabulary.py`, 2. `ABSATargetedDataset` — `dataloader.py`, 3. `TargetedABSAModel` — `model/targeted_absa.py`, 4. `AttentionPooling` — `model/attn_pooling.py`, 5. `ConditionalAttention` — `model/conditional_attn.py`, 6. `GatedFusion` — `model/gated_fusion.py`, 7. `MLPHead` — `model/mlp_head.py`, 8. `FocalLoss` — `train.py:30` (+2 more)

### Community 7 - "Community 7"
Cohesion: 0.22
Nodes (8): Data Processing, Dependencies (Not tracked in requirements file — inferred from imports), Dev Tooling, Frameworks & Libraries, Languages, ML/DL Frameworks, Runtime, Technology Stack

### Community 8 - "Community 8"
Cohesion: 0.29
Nodes (6): Data Sources, Datasets, Environment Configuration, External Services / APIs, Integrations, Model Checkpoints / Pretrained Models

### Community 9 - "Community 9"
Cohesion: 0.33
Nodes (4): Tensor, Mã hóa chuỗi văn bản đầu vào bằng PhoBERT Tokenizer.         Trả về một dictiona, Mã hóa cặp (sentence, target) cho mô hình targeted ABSA., Đánh dấu các token của target trong input pair.         Cách này không phụ thuộc

## Knowledge Gaps
- **41 isolated node(s):** `System Overview`, `Design Pattern`, `1. `Vocabulary` — `vocabulary.py``, `2. `ABSATargetedDataset` — `dataloader.py``, `3. `TargetedABSAModel` — `model/targeted_absa.py`` (+36 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Vocabulary` connect `Community 0` to `Community 9`, `Community 2`?**
  _High betweenness centrality (0.188) - this node is a cross-community bridge._
- **Why does `TargetedABSAModel` connect `Community 1` to `Community 2`?**
  _High betweenness centrality (0.113) - this node is a cross-community bridge._
- **Why does `ABSATrainer` connect `Community 2` to `Community 0`, `Community 1`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Vocabulary` (e.g. with `ABSATrainer` and `FocalLoss`) actually correct?**
  _`Vocabulary` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `TargetedABSAModel` (e.g. with `AttentionPooling` and `ConditionalAttention`) actually correct?**
  _`TargetedABSAModel` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `ABSATrainer` (e.g. with `TargetedABSAModel` and `Vocabulary`) actually correct?**
  _`ABSATrainer` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Dataset cho targeted ABSA.      Mỗi annotation [start, end, label] trong một rev`, `Collate batch gồm tensor và metadata string.     Tensor được stack; các field te`, `Hàm Loss linh hoạt dựa trên loss_key từ config` to the rest of the system?**
  _52 weakly-connected nodes found - possible documentation gaps or missing edges._