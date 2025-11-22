# Contributing Guidelines

Thank you for your interest in contributing to this project! This document provides guidelines and best practices for contributing.

## Development Process

### AI-Assisted Development with Human Oversight

This project embraces a modern development workflow:

1. **AI Bootstrapping**: Use AI coding tools to generate initial implementations
2. **Human Review**: Carefully review and validate all generated code
3. **Testing**: Verify results match the original paper
4. **Documentation**: Document implementation details and deviations

### Workflow Steps

1. **Choose a Paper**
   - Select an unimplemented paper from issues or propose a new one
   - Create an issue to claim the paper and avoid duplicate work

2. **Research Phase**
   - Read the paper thoroughly
   - Identify key algorithms and architectures
   - Note any ambiguities or implementation details

3. **Implementation Phase**
   - Generate initial code using AI assistance
   - Structure code in a clean, modular way
   - Add comprehensive type hints and docstrings
   - Include inline comments for complex logic

4. **Validation Phase**
   - Implement tests for core functionality
   - Compare results with paper benchmarks
   - Test edge cases and error handling

5. **Documentation Phase**
   - Write a detailed README for the implementation
   - Document hyperparameters and training procedures
   - Include usage examples
   - Add citation information

## Code Standards

### Python Style

- Follow PEP 8 guidelines
- Use `black` for formatting (line length: 88)
- Use `isort` for import sorting
- Use `flake8` for linting
- Use `mypy` for type checking

```bash
# Format code
black .
isort .

# Check style
flake8 .
mypy .
```

### Documentation

- All modules should have docstrings explaining their purpose
- All public functions and classes must have docstrings
- Use Google-style docstrings

```python
def train_model(model: nn.Module, data_loader: DataLoader, epochs: int = 10) -> Dict[str, float]:
    """Train a neural network model.
    
    Args:
        model: The neural network to train
        data_loader: DataLoader providing training data
        epochs: Number of training epochs (default: 10)
        
    Returns:
        Dictionary containing training metrics
        
    Raises:
        ValueError: If epochs is less than 1
    """
    pass
```

### Testing

- Write unit tests for all core functionality
- Aim for >80% code coverage
- Use pytest fixtures for common setup
- Test edge cases and error conditions

```python
def test_model_forward_pass():
    """Test that model produces correct output shape."""
    model = MyModel(input_dim=10, output_dim=5)
    x = torch.randn(32, 10)
    output = model(x)
    assert output.shape == (32, 5)
```

## Repository Structure

```
paper-name/
├── README.md              # Paper summary and usage
├── model.py               # Model architecture
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── config.yaml           # Configuration file
├── utils.py              # Helper functions
└── tests/
    ├── test_model.py
    └── test_utils.py
```

## Pull Request Process

1. **Branch Naming**
   - Feature: `feature/paper-name`
   - Bug fix: `fix/issue-description`
   - Documentation: `docs/description`

2. **Commit Messages**
   - Use clear, descriptive commit messages
   - Start with a verb: "Add", "Fix", "Update", "Refactor"
   - Reference issues: "Fix #123: Correct attention mask"

3. **PR Description**
   - Describe what the PR implements
   - Link to the paper (arXiv or venue)
   - Note any deviations from the paper
   - Include example usage
   - Add benchmark results if applicable

4. **Review Process**
   - At least one maintainer approval required
   - All tests must pass
   - Code coverage should not decrease
   - Documentation must be complete

## Human-in-the-Loop Validation

### What to Review

When reviewing AI-generated code:

- **Correctness**: Does it match the paper's algorithm?
- **Efficiency**: Are there performance bottlenecks?
- **Edge Cases**: Does it handle invalid inputs?
- **Reproducibility**: Are random seeds set appropriately?
- **Documentation**: Is the code well-explained?

### Common AI Code Issues

- Hallucinated APIs or functions
- Incorrect tensor dimensions
- Missing error handling
- Over-complicated logic
- Inadequate documentation

## Questions?

Feel free to open an issue or discussion if you have questions about contributing!
