# Contributing to THEMAP

Thank you for your interest in contributing to THEMAP! This document provides guidelines for contributing to the project.

## Getting Started

### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/yourusername/THEMAP.git
   cd THEMAP
   ```

2. **Set up development environment**:
   ```bash
   # Create conda environment
   conda env create -f environment.yml
   conda activate themap

   # Install in development mode with all dependencies
   pip install -e ".[dev,test,docs]"
   ```

3. **Verify installation**:
   ```bash
   # Run tests
   python run_tests.py

   # Check code formatting
   make lint

   # Build documentation
   make docs-build
   ```

## Development Workflow

### Code Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Add tests** for new functionality:
   ```bash
   # Add tests in tests/ directory
   # Run tests to ensure they pass
   python run_tests.py
   ```

4. **Update documentation**:
   ```bash
   # Add docstrings to new functions/classes
   # Update relevant docs in docs/
   # Build docs to check for issues
   make docs-build
   ```

5. **Format and lint code**:
   ```bash
   make format  # Auto-format with ruff
   make lint    # Check for issues
   ```

6. **Commit changes**:
   ```bash
   git add .
   git commit -m "feat: add new distance metric for proteins"
   ```

### Commit Message Convention

We follow conventional commit format:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions/modifications
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `ci:` - CI/CD changes

Examples:
```
feat: add sequence identity distance for proteins
fix: handle edge case in OTDD computation
docs: update API documentation for TaskDistance
test: add comprehensive tests for MoleculeDatasetDistance
```

### Pull Request Process

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create pull request** on GitHub with:
   - Clear description of changes
   - Link to related issues
   - Screenshots/examples if applicable

3. **Address review feedback**:
   - Respond to comments
   - Make requested changes
   - Update tests/docs as needed

4. **Ensure CI passes**:
   - All tests pass
   - Code coverage maintained
   - Documentation builds successfully

## Code Standards

### Python Style

- **PEP 8** compliance (enforced by ruff)
- **Type hints** for all public functions
- **Docstrings** for all public classes and functions
- **Maximum line length**: 110 characters

### Documentation Style

```python
def compute_distance(
    source_features: List[np.ndarray],
    target_features: List[np.ndarray],
    method: str = "euclidean"
) -> Dict[str, Dict[str, float]]:
    """Compute pairwise distances between feature sets.

    Args:
        source_features: List of feature arrays for source tasks
        target_features: List of feature arrays for target tasks
        method: Distance computation method ("euclidean", "cosine", "otdd")

    Returns:
        Dictionary with target task IDs as keys, containing dictionaries
        with source task IDs as keys and distance values.

    Raises:
        ValueError: If method is not supported
        DistanceComputationError: If computation fails

    Example:
        >>> source_features = [np.array([1, 2, 3])]
        >>> target_features = [np.array([4, 5, 6])]
        >>> distances = compute_distance(source_features, target_features)
        >>> print(distances)
        {'target_0': {'source_0': 5.196}}
    """
```

### Testing Standards

- **Comprehensive coverage**: Aim for >90% test coverage
- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test component interactions
- **Error handling**: Test exception cases
- **Mocking**: Use mocks for external dependencies

```python
def test_euclidean_distance_success(self, mock_compute_features, sample_tasks):
    """Test successful euclidean distance computation."""
    # Arrange
    mock_compute_features.return_value = sample_features
    distance = MoleculeDatasetDistance(tasks=sample_tasks, method="euclidean")

    # Act
    result = distance.euclidean_distance()

    # Assert
    assert isinstance(result, dict)
    assert "CHEMBL004" in result
    assert isinstance(result["CHEMBL004"]["CHEMBL001"], float)
```

## Types of Contributions

### Bug Reports

When reporting bugs, please include:

1. **Python version** and operating system
2. **THEMAP version** and installation method
3. **Minimal code example** that reproduces the issue
4. **Expected vs actual behavior**
5. **Error messages** and stack traces
6. **Environment details** (GPU, dependencies, etc.)

**Bug Report Template**:
```markdown
## Bug Description
Brief description of the issue.

## Environment
- Python version: 3.10.8
- THEMAP version: 0.1.0
- OS: Ubuntu 22.04
- GPU: NVIDIA RTX 4090 (optional)

## Code to Reproduce
```python
# Minimal example that reproduces the bug
from themap.distance import MoleculeDatasetDistance
# ... rest of code
```

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened, including error messages.

## Additional Context
Any other relevant information.
```

### Feature Requests

For new features, please:

1. **Check existing issues** to avoid duplicates
2. **Describe the use case** and motivation
3. **Propose an API design** if applicable
4. **Consider backward compatibility**
5. **Offer to implement** if you're able

### Documentation Improvements

Documentation contributions are highly valued:

- **Fix typos** and grammar errors
- **Add examples** and use cases
- **Improve API documentation**
- **Create tutorials** for new features
- **Update getting started guides**

### Performance Optimizations

When contributing performance improvements:

1. **Benchmark current performance**
2. **Profile the optimization**
3. **Ensure correctness** is maintained
4. **Add performance tests** if applicable
5. **Document performance characteristics**

## Review Process

### Code Review Criteria

Reviewers will check for:

- **Functionality**: Does the code work as intended?
- **Tests**: Are there adequate tests with good coverage?
- **Documentation**: Are docstrings and guides updated?
- **Style**: Does code follow project conventions?
- **Performance**: Are there any performance regressions?
- **Breaking changes**: Are they necessary and well-documented?

### Review Timeline

- **Initial response**: Within 3 business days
- **Full review**: Within 1 week for small changes, 2 weeks for large changes
- **Maintainer availability**: Best effort, but may be delayed during busy periods

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:

- **Be respectful** in all interactions
- **Be constructive** when providing feedback
- **Be collaborative** and help others learn
- **Be patient** with new contributors
- **Report inappropriate behavior** to maintainers

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests, discussions
- **Pull Requests**: Code contributions and reviews
- **GitHub Discussions**: General questions and community support

## Recognition

Contributors will be:

- **Listed in CONTRIBUTORS.md**
- **Credited in release notes** for significant contributions
- **Invited as collaborators** for ongoing contributors
- **Acknowledged in publications** that use their contributions

## Getting Help

If you need help:

1. **Check the documentation** first
2. **Search existing issues** for similar problems
3. **Ask in GitHub Discussions** for general questions
4. **Create an issue** for specific bugs or feature requests
5. **Reach out to maintainers** for sensitive issues

## Development Tips

### Useful Commands

```bash
# Run specific test file
pytest tests/distance/test_tasks_distance_comprehensive.py -v

# Run tests with coverage
python run_tests.py coverage

# Format code
make format

# Check types
mypy themap/

# Build and serve docs
make docs-serve

# Clean all artifacts
make clean
```

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debugger
import pdb; pdb.set_trace()

# Profile performance
import cProfile
cProfile.run('your_function()')
```

### Common Issues

1. **Import errors**: Check that all dependencies are installed
2. **Test failures**: Ensure you're using the correct Python version
3. **Memory issues**: Use smaller datasets for testing
4. **GPU issues**: Ensure CUDA is properly configured

Thank you for contributing to THEMAP! Your efforts help make computational drug discovery more accessible to researchers worldwide. 🧬💊🔬
