# S-XG-NID — Distilled

**Shrinking the semantic intrusion detector until it fits on the hardware that
actually sits on the network.**

The full S-XG-NID stack — sentence embeddings over packet payloads, a heterogeneous
graph neural network, gradient boosting, an explanation layer — assumes a machine
with a GPU. The devices that most need intrusion detection are IoT gateways with
neither. This repository is the teacher–student branch: train the large model,
then distil it into something small enough to run at the edge.

> **Status: early.** The graph builder, the model and the training loop exist and a
> test covers the builder. The distillation objective itself is still being brought
> up. Treat this as work in progress, not a finished result.

---

## What's here

```
src/s_xg_nid/
  train.py                 training loop over the heterogeneous graph
  models/gnn.py            XG_NID_Model
  graph/builder.py         constructs the HeteroData graph
  graph/structure_learning.py
scripts/
  download_data.py         fetch the datasets
  extract_iot_ips.py       isolate IoT device addresses from the captures
  inspect_datasets.py      sanity-check what was downloaded
tests/test_graph_builder.py
Papers/                    the seven primary sources this design is built on
S-XG-NID_Technical_Design_Document.docx
S-XG-NID_Extensions_Volume_II.docx
```

**Graph and sampling.** `train.py` loads a prebuilt `HeteroData` object and samples
manually rather than relying on a loader: edge indices are kept on the CPU for fast
slicing while node features sit on the GPU, and neighbourhoods are cut with
`k_hop_subgraph` / `subgraph` from PyTorch Geometric. The two edge types are
`('flow', 'targets', 'entity')` and `('entity', 'sends', 'flow')`.

**Reading first.** `Papers/` holds the seven papers the approach is derived from —
including work on self-knowledge distillation for lightweight IoT intrusion
detection and on hybrid GNNs over heterogeneous, dynamic network data. The design
document records which idea came from where.

## Running it

```bash
pip install torch torch-geometric numpy tqdm
python src/s_xg_nid/train.py
python -m pytest tests/
```

**Two things you must change first:**

1. `train.py` has a **hardcoded data path** (`base_path = "/media/mannava/D/..."`).
   Point it at your own `graph_object.pt`.
2. The committed config is a smoke-test config — `EPOCHS = 1`, `BATCH_SIZE = 1024`.
   Raise the epoch count before reading anything into the output.

## Related

- [`S-XG-NID`](https://github.com/Mannava-Daasaradhi/S-XG-NID) — the original.
- [`S-XG-NID-v2`](https://github.com/Mannava-Daasaradhi/S-XG-NID-v2) — the full
  seven-layer pipeline with the evidence-grounded explanation layer.
