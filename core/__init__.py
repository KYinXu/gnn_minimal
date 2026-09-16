"""
GNN-based Peptide MIC Classification

This module implements Graph Neural Networks for antimicrobial peptide (AMP)
classification using ESMFold-predicted 3D structures.

Architecture:
- Each peptide is represented as a graph
- Nodes = amino acid residues
- Edges = sequential bonds + spatial contacts (Cα-Cα < threshold)
- Node features = AA properties + pLDDT + structural features
- Edge features = distances + edge types

Models available:
- GCN: Graph Convolutional Network
- GAT: Graph Attention Network  
- EGNN: E(n) Equivariant Graph Neural Network
"""

import warnings
warnings.filterwarnings('ignore', message='.*Disabling its usage.*', category=UserWarning)
warnings.filterwarnings('ignore', message=".*torch-scatter.*was not found", category=UserWarning)

from .data_utils import (
    PeptideGraphDataset,
    pdb_to_graph,
    create_dataloaders,
    NODE_INPUT_DIM,
    NodeFeatureGroups,
    node_input_dim,
    node_feature_groups_from_cli,
    node_feature_groups_from_config_value,
    wants_esm2_residue_nodes,
)
from .models import GCN, GAT, EGNN, PeptideGNN
from .train import train_epoch, evaluate, run_training

__all__ = [
    'PeptideGraphDataset',
    'pdb_to_graph',
    'create_dataloaders',
    'NODE_INPUT_DIM',
    'NodeFeatureGroups',
    'node_input_dim',
    'node_feature_groups_from_cli',
    'node_feature_groups_from_config_value',
    'wants_esm2_residue_nodes',
    'GCN',
    'GAT',
    'EGNN',
    'PeptideGNN',
    'train_epoch',
    'evaluate',
    'run_training',
]
