# THEMAP Testing Guide

This directory contains comprehensive tests for the THEMAP project. The test suite covers all major components with unit tests, integration tests, and error handling scenarios.

## Quick Start

### Running Tests

```bash
# From project root directory

# Run all tests
python run_tests.py
# or
make test

# Run specific test categories
python run_tests.py unit           # Unit tests only
python run_tests.py integration    # Integration tests only
python run_tests.py distance       # Distance module tests only
python run_tests.py fast           # Exclude slow tests

# Run with coverage
python run_tests.py coverage
# or
make test-coverage
```

### Test Structure

```
tests/
├── conftest.py                                    # Shared fixtures and configuration
├── distance/
│   ├── __init__.py
│   ├── test_tasks_distance.py                     # Basic tests (legacy)
│   └── test_tasks_distance_comprehensive.py       # Comprehensive test suite
├── data/
│   └── test_*.py                                  # Data module tests
└── utils/
    └── test_*.py                                  # Utility module tests
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- Test individual functions and methods in isolation
- Use mocking extensively to avoid dependencies
- Fast execution (< 1 second per test)

### Integration Tests (`@pytest.mark.integration`)
- Test component interactions
- Use real data when possible
- May require external dependencies

### Slow Tests (`@pytest.mark.slow`)
- Tests that take significant time (> 5 seconds)
- Often involve large computations or model loading
- Can be skipped with `-m "not slow"`

## Distance Module Tests

The `test_tasks_distance_comprehensive.py` file provides extensive coverage for the distance computation module:

### Test Classes:
- `TestUtilityFunctions` - Tests for helper functions and validation
- `TestAbstractTasksDistance` - Tests for base class functionality
- `TestMoleculeDatasetDistance` - Tests for molecule distance computation
- `TestProteinDatasetDistance` - Tests for protein distance computation
- `TestTaskDistance` - Tests for combined task distance functionality
- `TestErrorHandling` - Tests for error scenarios and edge cases
- `TestIntegration` - End-to-end workflow tests

### Key Features Tested:
- ✅ Input validation and error handling
- ✅ Distance computation methods (OTDD, Euclidean, Cosine)
- ✅ Resource management and memory safety
- ✅ File I/O operations (loading/saving distances)
- ✅ GPU/CPU tensor handling
- ✅ Multi-modal distance combinations
- ✅ Legacy compatibility modes
- ✅ Logging and monitoring integration

## Running Specific Tests

### By File
```bash
pytest tests/distance/test_tasks_distance_comprehensive.py
```

### By Class
```bash
pytest tests/distance/test_tasks_distance_comprehensive.py::TestMoleculeDatasetDistance
```

### By Method
```bash
pytest tests/distance/test_tasks_distance_comprehensive.py::TestMoleculeDatasetDistance::test_euclidean_distance_success
```

### By Marker
```bash
pytest -m unit                    # Unit tests only
pytest -m "unit and not slow"     # Fast unit tests only
pytest -m integration             # Integration tests only
```

### With Keyword Filtering
```bash
pytest -k "distance"              # Tests with 'distance' in name
pytest -k "not gpu"               # Exclude GPU tests
```

## Test Data and Fixtures

The test suite uses extensive mocking to avoid dependencies on real data files. Key fixtures include:

- `sample_molecule_dataset` - Mock molecule dataset
- `sample_protein_dataset` - Mock protein dataset
- `sample_task` - Mock task with both data types
- `sample_tasks` - Mock Tasks collection
- `sample_features` - Pre-computed feature arrays

## Coverage

To generate coverage reports:

```bash
# Terminal report
pytest --cov=themap --cov-report=term-missing

# HTML report
pytest --cov=themap --cov-report=html
# Open htmlcov/index.html in browser

# Both
python run_tests.py coverage
```

## Performance Testing

For performance-sensitive tests:

```bash
# Show test durations
pytest --durations=10

# Profile with pytest-benchmark (if installed)
pytest --benchmark-only
```

## Debugging Tests

### Verbose Output
```bash
pytest -v                         # Verbose test names
pytest -vv                        # Very verbose output
pytest -s                         # Don't capture stdout
```

### Debug Failing Tests
```bash
pytest -x                         # Stop on first failure
pytest --tb=long                  # Long traceback format
pytest --pdb                      # Drop into debugger on failure
```

### Run Single Test with Debug
```bash
pytest tests/distance/test_tasks_distance_comprehensive.py::TestMoleculeDatasetDistance::test_euclidean_distance_success -v -s --tb=long
```

## Continuous Integration

The test configuration is optimized for CI environments:

- Warnings are filtered to reduce noise
- Short traceback format for cleaner output
- Strict marker enforcement prevents typos
- Parallel execution support with pytest-xdist

## Best Practices

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **Isolation**: Each test should be independent and not rely on others
3. **Mocking**: Use mocks for external dependencies to ensure reliability
4. **Assertions**: Include clear assertions with helpful error messages
5. **Markers**: Tag tests appropriately for easy filtering
6. **Documentation**: Include docstrings explaining complex test scenarios

## Contributing

When adding new tests:

1. Follow the existing naming convention (`test_*.py`)
2. Add appropriate markers (`@pytest.mark.unit`, etc.)
3. Include both success and failure scenarios
4. Mock external dependencies
5. Update this README if adding new test categories
