<p align="center">
  <img src="ensemble_logo.jpg" alt="Logo" width="400">
  <br /> <br / >
</p>

# NdLinear Is All You Need for Representation Learning

**Authors:**  Alex Reneau, Jerry Yao-Chieh Hu, Zhongfang Zhuang, Ting-Chun Liu  

## Overview

**NdLinear** ([paper link](https://arxiv.org/abs/2503.17353)) is an innovative linear transformation that preserves the multi-dimensional structure of data, enhancing both the representational power and parameter efficiency of neural networks. By operating along each dimension separately, it captures dependencies commonly overlooked by standard fully connected layers. NdLinear serves as an ideal foundational building block for large-scale models.


## Key Features

- **Structure Preservation:** Retains the original data format and shape.
- **Parameter Efficiency:** Reduces parameter count while improving performance.
- **Minimal Overhead:** Maintains the same complexity as conventional linear layers.
- **Easy Integration:** Seamlessly replaces existing linear layers with a multi-dimensional design.

## Community Engagement

We encourage the community to integrate **NdLinear** into their Hugging Face models, Kaggle projects, and GitHub repositories! By leveraging the multi-dimensional capabilities of NdLinear, you can enhance your machine learning workflows with improved representational power and efficiency.

### How to Get Involved

- **Hugging Face Models:** Easily enhance your Hugging Face transformers and other models by incorporating NdLinear layers. Share your models with the community on the [Hugging Face Model Hub](https://huggingface.co/models) and contribute to the open-source ecosystem.

- **Kaggle Projects:** Boost your Kaggle competition submissions by replacing standard linear layers with NdLinear in your codebase. Showcase the results with the Kaggle community by sharing your notebooks and findings.

- **GitHub Repositories:** Integrate NdLinear into your open-source projects. We welcome contributions and collaborations to further expand its applications and optimize performance across different domains.

### Share Your Work

If you use NdLinear in your projects, we would love to hear from you! Together, we can explore the vast potential and inspire innovation within the community. Join the conversation and start experimenting with NdLinear today!

## Installation

To integrate NdLinear into your projects, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/ensemble-core/NdLinear.git
cd NdLinear
uv sync
```

## Usage

NdLinear seamlessly integrates into various neural network architectures such as CNNs, RNNs, and Transformers.

### Example 1: Replacing a Standard Linear Layer with NdLinear

This example demonstrates how to replace a standard linear layer with an NdLinear layer to efficiently process multi-dimensional input data, such as a batch of images.

```python
import torch
from ndlinear import NdLinear

# Input data dimensions
input_tensor = torch.randn(32, 28, 28, 3)  # Example: batch of images (batch_size, height, width, channels)

# Using NdLinear
ndlinear_layer = NdLinear(input_dims=(28, 28, 3), hidden_size=(64, 64, 6))

output = ndlinear_layer(input_tensor)
```

### Example 2: Transformer

In transformer architectures, you might need to manipulate multi-dimensional tensors for efficient linear operations. Here's how you can use `NdLinear` with a 3D input tensor:

```python
import torch 
from ndlinear import NdLinear

input_tensor = torch.randn(32, 28, 28) 

# Reshape the input tensor for linear operations
input_tensor = input_tensor.reshape(-1, 28, 1)  # New shape: (batch_size * num_tokens, token_dim, 1)

# Define an NdLinear layer with suitable input and hidden dimensions
ndlinear_layer = NdLinear(input_dims=(28, 1), hidden_size=(32, 1))

# Perform the linear transformation
output = ndlinear_layer(input_tensor)

# Reshape back to the original dimensions after processing
output = output.reshape(batch_size, num_tokens, -1)  # Final output shape: (32, 28, 32)
```

This example illustrates how `NdLinear` can be integrated into transformer models by manipulating the tensor shape, thereby maintaining the structure necessary for further processing and achieving efficient projection capabilities.

### Example 3: Multilayer Perceptron 

This example demonstrates how to use the `NdLinear` layers in a forward pass setup, making integration into existing MLP structures simple and efficient.

```python 
import torch
from ndlinear import NdLinear

input_tensor = torch.randn(32, 128)

# Define the first NdLinear layer for the MLP with input dimensions (128, 8) and hidden size (64, 8)
layer1 = NdLinear(input_dims=(128, 8), hidden_size=(64, 8))

# Define the second NdLinear layer for the MLP with input dimensions (64, 8) and hidden size (10, 2)
layer2 = NdLinear(input_dims=(64, 8), hidden_size=(10, 2))

x = F.relu(layer1(input_tensor))

output = layer2(x)
```

### Example 4: Edge Case

When `input_dims` and `hidden_size` are one-dimensional, `NdLinear` functions as a conventional `nn.Linear` layer, serving as an edge case where `n=1`.

```python
from ndlinear import NdLinear

# Defining NdLinear with one-dimensional input and hidden sizes
layer1 = NdLinear(input_dims=(32,), hidden_size=(64,))
```

## Examples of Applications

We've tested NdLinear across various tasks and are excited to share some examples with you. More examples are on the way, so stay tuned!

- **Image Classification.** The `cnn_img_classification.py` script is designed for image classification using a CNN-based model on the CIFAR-10 dataset. You can execute it with the command: `python src/cnn_img_classification.py`.
- **Time Series Forecasting.** The `ts_forecast.py` script performs time-series forecasting on ETT datasets. You can run it using: `python src/ts_forecast.py`.
- **Text Classification.** The script `txt_classify_bert.py` utilizes BERT for text classification tasks. Start it with the command: `python src/txt_classify_bert.py`.
- **Vision Transformers.** The `vit_distill.py` script is set up for knowledge distillation using the Vision Transformer (ViT) model. Execute it with the command: `torchrun --nnodes 1 --nproc_per_node=4 src/vit_distill.py`.

## Citation

If you find NdLinear useful in your research, please cite our work:

```bibtex
@article{reneau2025ndlinear,
  title={NdLinear Is All You Need for Representation Learning},
  author={Reneau, Alex and Hu, Jerry Yao-Chieh and Zhuang, Zhongfang and Liu, Ting-Chun},
  journal={Ensemble AI},
  year={2025},
  note={\url{https://arxiv.org/abs/2503.17353}}
}
```

## Contact

For questions or collaborations, please reach out to [Alex Reneau](mailto:alex@ensemblecore.ai).

## License

This project is distributed under the Apache 2.0 license. You can view the [LICENSE](https://github.com/ensemble-core/NdLinear/blob/main/LICENSE) file for more details.