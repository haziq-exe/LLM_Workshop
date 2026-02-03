# LLM Workshop

Interactive demonstrations and hands-on code for understanding Large Language Models from the ground up. From Unlocking LLMs workshop hosted on 3rd February 2026 for Google Developers Group on Campus

Colab link: https://colab.research.google.com/drive/1IzYj4u_D4K4C7yW7K45Yhcpmru6gDeUw?usp=sharing

## Overview

This repository contains the supporting code for a workshop introducing core concepts in Large Language Models. The workshop is designed to build intuition through interactive demos rather than overwhelming learners with implementation details.

## Workshop Structure

The workshop covers eight fundamental topics:

1. **Neural Networks & Gradient Descent** - Visualizing how models learn through optimization
2. **Tokenization** - Understanding how text is converted to numerical representations
3. **Embeddings** - Exploring how meaning is encoded in vector space
4. **Attention Mechanisms** - Interactive exploration of transformer attention patterns
5. **Uncertainty Quantification** - Visualizing model confidence through token probabilities
6. **RAG** - Retrieval-Augmented Generation techniques
7. **Quantization** - Model compression methods
8. **Super Weights** - Identifying critical parameters in neural networks

## Repository Structure

```
├── PartI_NeuralNets/
│   ├── GradientDescentDemo.py
│   └── __init__.py
├── PartII_Tokenization/
│   ├── TokenizationDemo.py
│   ├── EmbeddingDemo.py
│   └── __init__.py
├── PartIII_Attention/
│   ├── AttentionDemo.py
│   └── __init__.py
├── PartV_Uncertainty/
│   ├── UncertaintyDemo.py
│   └── __init__.py
└── PartVIII_SuperWeight/
    └── SuperWeight.py
```

## Installation

```bash
pip install torch transformers matplotlib numpy scikit-learn ipywidgets
```

## Usage

The demos are designed to be imported and run directly in Jupyter notebooks:

```python
# Example: Tokenization Demo
from TokenizationDemo import TokenizationDemo
TokenizationDemo()  # Prompts for input and displays visualization
```

```python
# Example: Attention Visualization
from AttentionDemo import AttentionDemo
AttentionDemo()  # Interactive attention pattern explorer
```

Each demo class handles its own user interaction and visualization, keeping the notebook focused on concepts rather than code.

## Features

- **Interactive Visualizations**: All demos include matplotlib-based visualizations
- **Minimal Code in Notebooks**: Import and run approach keeps learning focused
- **Real Models**: Uses pretrained models from Hugging Face (BERT, GPT-2, Phi-3)
- **Hands-on Learning**: Prompts for user input to explore different examples


## Workshop Notebook

The main workshop notebook provides explanations and calls these demos at appropriate points. The notebook is designed to be followed sequentially, building understanding of how LLMs work from basic principles to advanced techniques.

Colab link: https://colab.research.google.com/drive/1IzYj4u_D4K4C7yW7K45Yhcpmru6gDeUw?usp=sharing