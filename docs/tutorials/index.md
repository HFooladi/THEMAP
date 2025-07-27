# Tutorials

Welcome to the THEMAP tutorials! These step-by-step guides will help you master task hardness estimation and distance computation for molecular activity prediction.

## Tutorial Overview

### 🚀 Beginner Tutorials

1. **[Getting Started](../user-guide/getting-started.md)** - Basic installation and first steps

## Interactive Notebooks

All tutorials are available as interactive Jupyter notebooks that you can run locally:

```bash
# Install notebook dependencies
pip install -e ".[dev]"

# Launch Jupyter Lab
jupyter lab docs/tutorials/
```


```python
# Download tutorial data (if not included)
from themap.utils import download_tutorial_data

download_tutorial_data("tutorials/data/")
```

### Sample Datasets

- **ChEMBL Bioactivity Data**: 10 training + 3 test tasks
- **Protein Sequences**: Target protein sequences for each task
- **Molecular Embeddings**: Pre-computed molecular features
- **Metadata**: Assay descriptions and experimental conditions

## Prerequisites

### Python Knowledge
- Basic Python programming
- Familiarity with NumPy and Pandas
- Optional: Jupyter notebook experience

### Domain Knowledge
- Basic understanding of molecular representations (SMILES, etc.)
- Familiarity with machine learning concepts
- Optional: Knowledge of protein sequences and drug discovery

### Computational Resources

Most tutorials can run on:
- **CPU**: Standard laptop/desktop (8GB+ RAM recommended)
- **GPU**: Optional, speeds up OTDD computations
- **Storage**: ~1GB for tutorial data and caches

## Getting Help

### Tutorial Support

If you encounter issues with tutorials:

1. **Check Prerequisites**: Ensure all dependencies are installed
2. **Verify Data**: Confirm tutorial data is properly downloaded
3. **Read Error Messages**: THEMAP provides detailed error information
4. **Ask Questions**: Open an issue on [GitHub](https://github.com/HFooladi/THEMAP/issues)

### Common Issues

```python
# Installation issues
pip install -e ".[all]"  # Install all dependencies

# Memory issues
# Use smaller datasets or batch processing

# GPU issues
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Community

- **GitHub Discussions**: Share experiences and ask questions
- **Issues**: Report bugs or request features
- **Contributions**: Submit improvements to tutorials

## Contributing to Tutorials

We welcome contributions! To add or improve tutorials:

1. **Fork the repository**
2. **Create tutorial content** in Markdown and/or Jupyter format
3. **Test thoroughly** with different environments
4. **Submit pull request** with clear description

### Tutorial Guidelines

- **Clear objectives**: State what readers will learn
- **Step-by-step**: Break complex tasks into manageable steps
- **Code examples**: Include runnable code snippets
- **Error handling**: Show how to handle common issues
- **Real data**: Use realistic examples when possible

## What's Next?

Ready to get started? Here are recommended next steps:

- **New to THEMAP?** → [Getting Started](../user-guide/getting-started.md)

Happy learning! 🧪🔬📊
