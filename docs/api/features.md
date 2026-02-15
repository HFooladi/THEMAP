# Features Module

The features module provides unified feature extraction for molecules and proteins.

## MoleculeFeaturizer

::: themap.features.molecule.MoleculeFeaturizer
    options:
      show_root_heading: true
      heading_level: 3

### Available Featurizers

#### Fingerprints

| Featurizer | Description | Dimensions |
|------------|-------------|------------|
| `ecfp` | Extended Connectivity Fingerprints | 2048 |
| `fcfp` | Functional Connectivity Fingerprints | 2048 |
| `maccs` | MACCS Structural Keys | 167 |
| `topological` | Topological Fingerprints | 2048 |
| `avalon` | Avalon Fingerprints | 512 |
| `atompair` | Atom Pair Fingerprints | 2048 |
| `pattern` | Pattern Fingerprints | 2048 |
| `layered` | Layered Fingerprints | 2048 |
| `secfp` | SMILES Extended Connectivity Fingerprints | 2048 |
| `erg` | Extended Reduced Graph | varies |
| `estate` | E-State Fingerprints | 79 |
| `rdkit` | RDKit Fingerprints | 2048 |

Count variants (`ecfp-count`, `fcfp-count`, etc.) are also available.

#### Descriptors

| Featurizer | Description | Dimensions |
|------------|-------------|------------|
| `desc2D` | 2D Molecular Descriptors | ~200 |
| `mordred` | Mordred Descriptors | ~1600 |
| `cats2D` | CATS2D Pharmacophore | varies |
| `pharm2D` | 2D Pharmacophore | varies |
| `scaffoldkeys` | Scaffold Keys | varies |

#### Neural Embeddings

| Featurizer | Description | Dimensions |
|------------|-------------|------------|
| `ChemBERTa-77M-MLM` | ChemBERTa masked language model | 384 |
| `ChemBERTa-77M-MTR` | ChemBERTa multi-task regression | 384 |
| `MolT5` | Molecular T5 embeddings | 768 |
| `Roberta-Zinc480M-102M` | RoBERTa trained on ZINC | 768 |
| `gin_supervised_infomax` | GIN infomax embeddings | 300 |
| `gin_supervised_contextpred` | GIN context prediction | 300 |
| `gin_supervised_edgepred` | GIN edge prediction | 300 |
| `gin_supervised_masking` | GIN masking embeddings | 300 |

Run `themap list-featurizers` to see all available featurizers.

## ProteinFeaturizer

::: themap.features.protein.ProteinFeaturizer
    options:
      show_root_heading: true
      heading_level: 3

### Available Models

#### ESM2 Models

| Model | Parameters | Layers | Embedding Dim |
|-------|------------|--------|---------------|
| `esm2_t6_8M_UR50D` | 8M | 6 | 320 |
| `esm2_t12_35M_UR50D` | 35M | 12 | 480 |
| `esm2_t30_150M_UR50D` | 150M | 30 | 640 |
| `esm2_t33_650M_UR50D` | 650M | 33 | 1280 |

#### ESM3 Models

| Model | Description |
|-------|-------------|
| `esm3_sm_open_v1` | ESM3 small open model |

## FeatureCache

::: themap.features.cache.FeatureCache
    options:
      show_root_heading: true
      heading_level: 3

See the [Performance Optimization tutorial](../tutorials/performance-optimization.md) for caching and parallelism tips.
