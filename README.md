# 💎 Diamond Price Prediction using Deep Learning

A comprehensive machine learning pipeline that predicts diamond prices by combining vision-based models for feature extraction with tabular models for price prediction.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#️-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Models](#-models)
- [Results](#-results)
- [Development](#️-development)
- [Limitations and Known Issues](#️-limitations-and-known-issues)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#-license)
- [Authors](#-authors)
- [Acknowledgments](#-acknowledgments)
- [References](#-references)

## 🎯 Overview

This project implements a **two-stage modular pipeline** for estimating diamond prices from product images, addressing the challenge of predicting market value when only visual information is available and structured grading reports may be missing or unreliable.

**The Challenge**: Direct image→price prediction suffers from insufficient data (~1,400 samples), while our decomposed approach leverages 1-2 orders of magnitude more data (~60,000 for vision attributes and ~200,000 for price prediction), enabling deeper learning for each task.

### Why This Approach Works

The two-stage design separates **perception** (attribute prediction) from **valuation** (price regression):
1. **Stage 1**: Vision models predict grading attributes from images
2. **Stage 2**: Tabular models estimate price from predicted attributes + confidence features

This decomposition improves **interpretability**, allows each stage to exploit appropriate **inductive biases** for its modality, and enables systematic debugging and iteration.

### Key Highlights

- **Vision Models**: Extract 7 key diamond features from images (shape, cut, color, clarity, polish, symmetry, carat)
- **Tabular Models**: XGBoost and MLP models with Optuna hyperparameter tuning
- **End-to-End Pipeline**: Automated workflow from raw image to price estimate with uncertainty quantification
- **Strong Performance**: RMSE of ~$2,182-$2,195 and MAPE of ~10.5% on tabular data; end-to-end RMSE of $441 with inflation adjustment
- **Real-World Oriented**: Designed for marketplace scenarios where structured grading data is unavailable

## ✨ Features

- 🖼️ **Image-Based Feature Extraction**: Automated extraction of diamond characteristics from photographs
- 🤖 **Multiple Model Types**: CNN-based classifiers and regressors for different diamond attributes
- 📊 **Ensemble Approach**: Combines vision model outputs with tabular models for robust predictions
- 🎨 **Support for Multiple Diamond Shapes**: Round, cushion, emerald, oval, radiant, and heart
- 📈 **Performance Tracking**: Comprehensive evaluation metrics and visualization tools
- 🔧 **Flexible Pipeline**: Modular design allows easy customization and experimentation

## 🏗️ Architecture

The system follows a **two-stage modular architecture**:

```
Image I → Vision Models (fθ) → Predicted Attributes â → Tabular Models (gφ) → Price ŷ
```

### Stage 1: Vision Attribute Prediction

Each diamond attribute is predicted by an **independently trained single-task model** (multi-task learning was explored but didn't yield stable improvements):

#### Predicted Attributes
- **Shape**: Multi-class classification (Round, Princess, Emerald, Cushion, Oval, Radiant, Heart)
- **Cut**: Multi-class classification (Ideal, Excellent, Very Good, Good, Fair)
- **Color**: Tier-7 ordinal-aware classification (D-Z grades, with ordinal structure exploited)
- **Clarity**: Multi-class classification with ordinal regularization (IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1)
- **Polish**: Classification (Excellent, Very Good, Good; binary EX vs. rest in some experiments)
- **Symmetry**: Multi-class classification (Excellent, Very Good, Good)
- **Carat**: Regression with log(1 + carat) transform

**Note**: Fluorescence is explicitly **excluded** from the final pipeline to reduce scope and avoid relying on a visually ambiguous attribute under uncontrolled web-photo conditions.

#### Technical Implementation
- **Backbones**: ImageNet-pretrained CNNs (ResNet-18, ConvNeXt-Tiny)
- **Optimization**: Mixed precision (AMP), gradient clipping, cosine LR schedules with warmup
- **Stability**: Exponential moving average (EMA) of weights for stable validation
- **Class Imbalance**: WeightedRandomSampler, balanced accuracy metrics
- **Losses**: Cross-entropy for classification, SmoothL1 (Huber) for carat regression

#### Exported Features
For each image, the vision stage exports:
- Hard predictions (argmax classes)
- Softmax probability vectors
- Confidence measures (max probability, entropy, margin)
- Optional penultimate-layer embeddings

### Stage 2: Tabular Price Regression

#### Feature Construction
Tabular features built from vision outputs:
- One-hot encodings of predicted classes
- Probability vectors from softmax
- Confidence features (max prob, entropy, margins)
- Predicted carat value
- All numeric features standardized

#### Models
- **XGBoost**: Gradient-boosted decision trees (strong tabular baseline)
- **MLP**: Multi-layer perceptron with dropout and batch normalization
- **Ridge Regression**: Linear baseline for comparison

#### Training Strategy
- **Target Transform**: Log-space training: `z = log(y + ε)`, prediction: `ŷ = exp(ẑ) - ε`
- **Hyperparameter Tuning**: Optuna optimization (depth, learning rate, regularization)
- **Validation Metric**: RMSE in log-space
- **Evaluation Metrics**: RMSE (real $), MAE, MAPE, R²

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/AmitosForever/DL-Project.git
cd DL-Project
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Packages

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
xgboost>=2.0.0
optuna>=3.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
opencv-python>=4.8.0
tqdm>=4.65.0
kagglehub>=0.2.0
```

## 📖 Usage

### Quick Start

Follow these steps to run the complete pipeline:

#### Step 1: Train Vision Models

```bash
# Navigate to vision model directory
cd src/vision_model/final_version_all

# Train all vision models
python carat.py
python color.py
python clarity.py
python cut.py
python polish.py
python symmetry.py
python shape.py
```

Models will be saved to `src/combined_models/vision_models/`

#### Step 2: Prepare Your Dataset

```bash
# Use the provided notebook to prepare your image data
jupyter notebook src/combined_models/test_dataset.ipynb
```

Or use the existing diamond datasets in `src/combined_models/data/Diamonds/`

#### Step 3: Run Vision Models on Images

```bash
cd src/combined_models
python small_dataset_run.py
```

This will:
- Load trained vision models
- Process your images
- Save predictions to CSV/XLSX format

#### Step 4: Generate Final Price Predictions

```bash
# Open the combined models notebook
jupyter notebook combined_models.ipynb
```

Execute all cells to:
- Load vision model predictions
- Train/load tabular models
- Generate final price predictions
- Evaluate performance

### Training Individual Models

#### Vision Model Training Example
```python
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader

# Load your custom training script
# Example: Training color prediction model
python src/vision_model/color/diamond_color_final.py
```

#### Tabular Model Training Example
```python
# XGBoost training
jupyter notebook src/tabular_model/tree_model/xgboost.ipynb

# MLP training
jupyter notebook src/tabular_model/mlp/mlp.ipynb
```

## 📁 Project Structure

```
DL-Project/
│
├── src/
│   ├── combined_models/          # Main pipeline integration
│   │   ├── combined_models.ipynb # Final prediction notebook
│   │   ├── small_dataset_run.py  # Vision model inference
│   │   ├── test_dataset.ipynb    # Dataset preparation
│   │   ├── vision_models/        # Trained vision model checkpoints
│   │   └── data/                 # Diamond datasets
│   │       └── Diamonds/         # Shape-specific data
│   │           ├── round/
│   │           ├── cushion/
│   │           ├── emerald/
│   │           ├── oval/
│   │           ├── radiant/
│   │           └── heart/
│   │
│   ├── vision_model/             # Computer vision models
│   │   ├── final_version_all/   # Production-ready models
│   │   │   ├── carat.py
│   │   │   ├── color.py
│   │   │   ├── clarity.py
│   │   │   ├── cut.py
│   │   │   ├── polish.py
│   │   │   ├── symmetry.py
│   │   │   └── shape.py
│   │   ├── carat/               # Carat prediction experiments
│   │   ├── color/               # Color grading experiments
│   │   ├── clarity/             # Clarity grading experiments
│   │   ├── cut/                 # Cut quality experiments
│   │   ├── polish/              # Polish quality experiments
│   │   ├── symmetry/            # Symmetry experiments
│   │   ├── fluorescence/        # Fluorescence experiments
│   │   └── shape/               # Shape classification experiments
│   │
│   └── tabular_model/            # Price prediction models
│       ├── tree_model/           # XGBoost implementation
│       │   ├── xgboost.ipynb
│       │   └── results/
│       ├── mlp/                  # Neural network implementation
│       │   ├── mlp.ipynb
│       │   ├── optuna.ipynb     # Hyperparameter optimization
│       │   └── results/
│       ├── linear_model/         # Baseline models
│       │   ├── linear_regression.ipynb
│       │   └── result/
│       ├── models/               # Saved model checkpoints
│       │   ├── best_diamond_mlp_model.pth
│       │   ├── diamond_xgb_model.json
│       │   └── models_eval.ipynb
│       └── data/
│           └── data_prep.ipynb
│
├── requirements.txt
├── LICENSE
└── README.md
```

## 🤖 Models

### Vision Models Performance

All models trained on web-scraped diamond images with ImageNet-pretrained backbones.

| Attribute | Model Type | Accuracy | Close-Class Accuracy* | Notes |
|-----------|-----------|----------|----------------------|-------|
| **Shape** | CNN Classifier | **99.9%** | - | Near-perfect classification |
| **Color** | Tier-7 Ordinal CNN | **77.8%** | **96.9%** | Ordinal structure exploited |
| **Clarity** | Ordinal CNN | **54.5%** | **91.3%** | High close-class accuracy |
| **Cut** | CNN Classifier | **80.1%** | - | Multi-class classification |
| **Symmetry** | CNN Classifier | **85.2%** | - | Quality grading |
| **Polish** | CNN Classifier | **92.3%** | - | Binary/multi-class setup |
| **Carat** | CNN Regressor | RMSE: **0.240** | - | Log-transformed SmoothL1 loss |

**\*Close-Class Accuracy**: For ordinal attributes (color, clarity), this measures predictions within ±1 grade of ground truth. Most errors are adjacent-grade confusions rather than distant misclassifications, which is acceptable in practice since small ordinal deviations have limited price impact.

### Tabular Models Performance

Performance on **~200,000 samples** with ground-truth attributes → price:

| Model | RMSE (real $) | MAPE | Training Notes |
|-------|---------------|------|----------------|
| **Linear Regression** | $52,119,177 | 689.10% | Baseline (poor fit) |
| **Ridge Regression** | $52,124,590 | 689.13% | Regularized baseline |
| **MLP + Optuna** | **$2,195** | **10.56%** | Deep model with tuning |
| **XGBoost + Optuna** | **$2,182** | **10.50%** | Best tabular model |

**Key Observations**:
- Linear models fail completely on this heavy-tailed distribution
- Tree-based (XGBoost) slightly outperforms MLP (~10.50% vs 10.56% MAPE)
- Log-space training essential for handling price distribution skew

### End-to-End Image→Price Performance

Performance on **~1,400 image-price pairs** (full pipeline, no ground-truth attributes):

| Setup | RMSE | MAE | MAPE | R² | Notes |
|-------|------|-----|------|----|-------|
| **MLP (inflation-adjusted)** | **$441** | **$298** | **20.17%** | **0.44** | Recommended approach |
| **MLP (raw prices)** | $703 | $524 | 48.17% | -1.77 | Severe temporal drift |

**Critical Finding**: **Inflation adjustment is essential** for end-to-end image→price prediction. Without it, temporal price drift dominates the learning signal and performance degrades dramatically (negative R²).

## 📊 Results

### Vision Model Outputs

The vision models generate predictions for all diamond attributes, which are saved in structured format:

```csv
image_id, carat_pred, color_pred, clarity_pred, cut_pred, polish_pred, symmetry_pred, shape_pred
img_001.jpg, 1.25, E, VS1, Ideal, Excellent, Excellent, Round
img_002.jpg, 0.95, G, VVS2, Very Good, Excellent, Very Good, Princess
...
```

### Tabular Regression Results

Below are actual vs. predicted plots from the development experiments:

#### Evolution of Model Performance

1. **Linear/Ridge Regression** (Baseline)
   - RMSE: ~$52M, MAPE: ~689%
   - Complete failure on heavy-tailed distribution
   - Demonstrates necessity of nonlinear models and log-transform

2. **MLP with Optuna Tuning**
   - RMSE: $2,195, MAPE: 10.56%
   - Substantial improvement with deep architecture
   - Hyperparameter tuning critical for stability

3. **XGBoost with Optuna Tuning** (Best)
   - RMSE: $2,182, MAPE: 10.50%
   - Slightly outperforms MLP (consistent with tabular data literature)
   - Better handling of feature interactions and missing patterns

#### Key Observations

- **Log-space training essential**: Without log-transform, models fail to learn meaningful patterns
- **Tree-based strength**: XGBoost's slight edge validates findings that GBDTs excel on tabular data
- **MAPE interpretation**: ~10.5% error means $1,000 diamond typically predicted within ±$105
- **Tight diagonal alignment**: Scatter plots show strong correlation between actual and predicted prices

### End-to-End Pipeline Results

Full **image→price** evaluation on ~1,400 samples:

#### With Inflation Adjustment ✅ (Recommended)
- **RMSE**: $441.32
- **MAE**: $297.63  
- **MAPE**: 20.17%
- **R²**: 0.44

Strong alignment with ideal prediction line on log-log plots. Inflation normalization successfully removes temporal drift, enabling vision features to explain ~44% of price variance.

#### Without Inflation Adjustment ❌ (Not Recommended)
- **RMSE**: $703.46
- **MAE**: $524.21
- **MAPE**: 48.17%
- **R²**: -1.77 (worse than constant prediction)

Large dispersion and poor alignment. Temporal price drift dominates learning signal, making the problem nearly unsolvable from visual features alone.

**Critical Takeaway**: When performing end-to-end image→price prediction on temporally heterogeneous marketplace data, **inflation adjustment is non-negotiable**. Without it, macroeconomic factors overwhelm visual information.


### Dataset Characteristics

- **Vision Training**: ~60,000 images with attribute labels
- **Tabular Training**: ~200,000 samples with attributes + prices
- **End-to-End**: ~1,400 image-price pairs
- **Splits**: Listing-level stratification to reduce near-duplicate leakage
- **Source**: Publicly available web-scraped marketplace data

## 🛠️ Development

### Adding New Vision Models

1. Create a new Python file in the appropriate attribute directory
2. Implement the model architecture using PyTorch
3. Add training loop with proper evaluation metrics
4. Save the best model checkpoint
5. Update the pipeline to include your model

### Extending Tabular Models

1. Navigate to `src/tabular_model/`
2. Create a new notebook or script
3. Implement your model using the existing data loaders
4. Compare performance against baseline models
5. Update the ensemble if performance improves

## ⚠️ Limitations and Known Issues

### Current Limitations

1. **Dataset Noise**: Web-scraped prices contain market confounders (seller markup, promotional pricing) not observable from images
2. **Photographic Variability**: Uncontrolled lighting and camera settings introduce systematic bias in color/clarity predictions
3. **Error Propagation**: Vision stage errors compound in pricing stage; confidence features help but don't eliminate this
4. **Temporal Drift**: Prices vary over time due to inflation and market dynamics; inflation adjustment is essential
5. **Heavy-Tailed Distribution**: Extreme high-value diamonds may be underrepresented, affecting tail performance

### Use Case Considerations

This system is designed as **decision support** when structured grading data is unavailable. It should **NOT** be used as:
- An authoritative appraisal or replacement for certified grading reports
- Automated pricing without human oversight
- A guarantee of market value (estimates can be wrong due to non-visual factors)

### Stakeholders and Responsible Use

**For Consumers**:
- Estimates are approximate and depend on image quality
- Tool should be treated as guidance, not a guarantee
- Lighting, editing, and seller-driven factors can affect accuracy

**For Sellers/Marketplaces**:
- Model is decision support and may embed dataset biases
- Should not automatically set prices without human review
- Regular audits recommended across price bands and categories

**For Regulators**:
- System limitations, data sources, and failure modes are documented
- Uncertainty handling and confidence measures are provided
- Ongoing monitoring for systematic mispricing is recommended

## 🔮 Future Work

### Short-Term Improvements
- **Cleaner supervision**: Multi-view images, certified grading reports
- **Uncertainty quantification**: Calibration, ensemble disagreement, prediction intervals
- **Better confounders**: Incorporate seller/domain metadata, detect anomalies

### Medium-Term Research
- **Stronger tabular models**: Modern deep tabular architectures (TabularFM, FT-Transformer)
- **Semi-supervised learning**: Meta pseudo-labeling on small image→price dataset
- **Multi-task learning**: Revisit with improved regularization and task balancing
- **Domain adaptation**: Handle distribution shift between professional and user-generated photos

### Long-Term Vision
- **Market-aware modeling**: Time-series price trends, economic indicators
- **Explainability**: Attribution maps for vision models, feature importance for tabular
- **Active learning**: Intelligently query uncertain samples for human annotation
- **Production deployment**: Real-time inference API, model monitoring infrastructure

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Nadav Offir**
- **Amit Rubinshtein** - [GitHub](https://github.com/AmitosForever)

*Deep Learning Course Project*

## 🙏 Acknowledgments

- Deep Learning course instructors and TAs
- Authors of pretrained models: ResNet ([He et al., 2016](https://arxiv.org/abs/1512.03385)), ConvNeXt ([Liu et al., 2022](https://arxiv.org/abs/2201.03545))
- XGBoost and Optuna development teams
- PyTorch and scikit-learn communities
- Diamond dataset providers and marketplace platforms

## 📖 References

Key papers that influenced this work:

1. **Vision Backbones**: He et al. (2016) - ResNet; Liu et al. (2022) - ConvNeXt
2. **Tabular Learning**: Grinsztajn et al. (2022) - ["Why do tree-based models still outperform deep learning on typical tabular data?"](https://arxiv.org/abs/2207.08815)
3. **Deep Tabular Methods**: Gorishniy et al. (2021) - ["Revisiting Deep Learning Models for Tabular Data"](https://arxiv.org/abs/2106.11959)
4. **Gradient Boosting**: Chen & Guestrin (2016) - XGBoost
5. **Tabular Architectures**: Huang et al. (2020) - TabTransformer; Arik & Pfister (2019) - TabNet

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub or contact the repository maintainer.

---

⭐ If you find this project useful, please consider giving it a star!
