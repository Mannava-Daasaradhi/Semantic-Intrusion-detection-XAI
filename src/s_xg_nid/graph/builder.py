import pandas as pd
import torch
from torch_geometric.data import HeteroData
import os
import logging
import numpy as np

# --- CONFIG ---
# If you run out of RAM, set this to True to process in chunks (slower but safer)
# For 32GB RAM, False is fine.
LOW_MEMORY_MODE = False
# --------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GraphBuilder:
    def __init__(self, base_path):
        self.base_path = base_path
        self.parquet_path = os.path.join(base_path, "unified_dataset_final.parquet")
        
    def build(self):
        logging.info("Reading Parquet file...")
        df = pd.read_parquet(self.parquet_path)
        
        logging.info(f"Loaded {len(df)} rows. Mapping Entities to IDs...")
        
        # 1. Unique Entity Mapping (String IP -> Int ID)
        # We concat src and dst to find ALL unique entities in the network
        unique_entities = pd.unique(df[['src_entity', 'dst_entity']].values.ravel('K'))
        logging.info(f"Found {len(unique_entities)} Unique Network Entities (IPs/Devices).")
        
        # Create a mapping dictionary (String -> Int)
        entity_map = {name: i for i, name in enumerate(unique_entities)}
        
        # 2. Map the DataFrame columns to Integers
        logging.info("Encoding Source and Destination IDs...")
        src_nodes = df['src_entity'].map(entity_map).astype(np.int64).values
        dst_nodes = df['dst_entity'].map(entity_map).astype(np.int64).values
        
        # Flow IDs are just the row indices (0 to 49.5M)
        flow_nodes = np.arange(len(df), dtype=np.int64)
        
        # 3. Create PyTorch Geometric Data Object
        logging.info("Constructing HeteroData Tensors...")
        data = HeteroData()
        
        # --- NODE FEATURES ---
        # Entity Nodes: They are just "Identity" nodes, they don't have features yet.
        # We give them a dummy feature (1) or an embedding later.
        data['entity'].num_nodes = len(unique_entities)
        
        # Flow Nodes: They hold the actual stats
        feature_cols = ['byte_rate', 'packet_rate', 'syn_ratio', 'ack_ratio', 'fin_ratio', 
                        'protocol_tcp', 'protocol_udp', 'protocol_icmp']
        
        # Convert to Float32 Tensor
        x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        data['flow'].x = x
        
        # Labels (Attack vs Benign)
        y = torch.tensor(df['label'].values, dtype=torch.long)
        data['flow'].y = y
        
        # --- EDGES (The Connectivity) ---
        # Edge Type 1: Entity -> sends -> Flow
        # Format: [2, Num_Edges] where row 0 is Source, row 1 is Target
        edge_index_sends = torch.tensor([src_nodes, flow_nodes], dtype=torch.long)
        data['entity', 'sends', 'flow'].edge_index = edge_index_sends
        
        # Edge Type 2: Flow -> targets -> Entity
        edge_index_targets = torch.tensor([flow_nodes, dst_nodes], dtype=torch.long)
        data['flow', 'targets', 'entity'].edge_index = edge_index_targets
        
        # 4. Save
        output_path = os.path.join(self.base_path, "graph_object.pt")
        logging.info(f"Graph Construction Complete. Saving to {output_path}...")
        torch.save(data, output_path)
        
        # Save mapping for reverse-lookup later (Model Inference)
        # We verify checking mapping size first
        if len(unique_entities) < 10_000_000:
            map_path = os.path.join(self.base_path, "entity_mapping.npy")
            np.save(map_path, unique_entities)
            logging.info("Saved Entity Mapping for future lookups.")
            
        print("\n✅ GRAPH BUILT SUCCESSFULLY.")
        print(data)

if __name__ == "__main__":
    base_dir = "/media/mannava/D/S-XG-NID/data/01_raw"
    builder = GraphBuilder(base_dir)
    builder.build()