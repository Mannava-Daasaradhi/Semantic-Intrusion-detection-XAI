import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear

class XG_NID_Model(torch.nn.Module):
    def __init__(self, metadata, num_flow_features, num_entities, hidden_channels=64, out_channels=2):
        super().__init__()
        
        # 1. Entity Embedding
        # Entities (IPs) have no features, so we learn a vector for each unique IP.
        # This allows the model to remember "Bad IPs" across flows.
        self.entity_emb = nn.Embedding(num_entities, hidden_channels)
        
        # 2. Graph Convolution Layers (The "Reasoning" Layers)
        # Layer 1: Aggregates info from immediate neighbors
        self.conv1 = HeteroConv({
            # Flow -> targets -> Entity: Update Entity based on flows it receives
            ('flow', 'targets', 'entity'): SAGEConv((-1, -1), hidden_channels),
            # Entity -> sends -> Flow: Update Flow based on who sent it
            ('entity', 'sends', 'flow'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')

        # Layer 2: Aggregates info from 2-hop neighbors (Deep reasoning)
        self.conv2 = HeteroConv({
            ('flow', 'targets', 'entity'): SAGEConv((-1, -1), hidden_channels),
            ('entity', 'sends', 'flow'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')

        # 3. Input Projection for Flows
        # Projects the 8 statistical features up to 'hidden_channels' size
        self.flow_lin = Linear(num_flow_features, hidden_channels)

        # 4. Final Classifier (Predicts Attack vs Benign)
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # x_dict contains features for 'flow'. 
        # For 'entity', we must generate embeddings on the fly.
        
        # 1. Prepare Initial Embeddings
        # Flow features projected to hidden dim
        x_flow = self.flow_lin(x_dict['flow'])
        
        # Entity embeddings looked up from the learnable table
        # We assume the loader passes 'n_id' (node indices) for entities in the batch
        entity_ids = x_dict['entity'] # These are the IDs (0 to 323k)
        x_entity = self.entity_emb(entity_ids)
        
        # Update the dictionary with aligned features
        x_dict_calc = {
            'flow': x_flow,
            'entity': x_entity
        }

        # 2. Message Passing Layer 1
        x_dict_calc = self.conv1(x_dict_calc, edge_index_dict)
        x_dict_calc = {key: F.relu(x) for key, x in x_dict_calc.items()}
        x_dict_calc = {key: F.dropout(x, p=0.2, training=self.training) for key, x in x_dict_calc.items()}

        # 3. Message Passing Layer 2
        x_dict_calc = self.conv2(x_dict_calc, edge_index_dict)
        x_dict_calc = {key: F.relu(x) for key, x in x_dict_calc.items()}

        # 4. Classification
        # We only care about classifying FLOWS, not Entities.
        out = self.classifier(x_dict_calc['flow'])
        
        return out