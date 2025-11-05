# Fake News Detection using BERT

A comprehensive fake news detection system built with BERT (Bidirectional Encoder Representations from Transformers) that includes model training, evaluation, explainability analysis using SHAP, and robustness testing across multiple datasets.

## Overview

This project implements a binary classification system to distinguish between true and fake news articles. It features:

- **BERT-based classification** using `bert-base-uncased`
- **Multi-phase approach**: Initial training, evaluation, transfer learning, and model refinement
- **Explainability**: SHAP (SHapley Additive exPlanations) integration for model interpretability
- **Robustness testing**: Cross-dataset validation using GossipCop and PolitiFact datasets
- **Comprehensive metrics**: Accuracy, precision, recall, F1-score, and confusion matrix analysis

## Features

### Phase 1: Data Acquisition & Preprocessing
- Loads and consolidates True.csv and Fake.csv datasets
- Handles missing data with fallback synthetic examples
- Creates stratified train/validation/test splits (80/10/10)
- Combines article titles and text for comprehensive analysis

### Phase 2: BERT Model Training
- Utilizes pre-trained BERT model with fine-tuning
- Custom PyTorch Dataset implementation for efficient data loading
- Configurable hyperparameters (batch size, epochs, max sequence length)
- GPU acceleration support with automatic CPU fallback

### Phase 3: Model Evaluation
- Detailed performance metrics on test set
- Confusion matrix visualization with Type I/II error analysis
- Classification breakdown for true negatives, true positives, false positives, and false negatives

### Phase 4: SHAP Explainability
- Word-level contribution analysis
- Visual interpretation of model decisions
- Sample-by-sample prediction explanations
- Color-coded feature importance (red for fake indicators, blue for true indicators)

### Phase 5: Transfer Learning Validation
- Cross-dataset testing on FakeNewsNet (GossipCop + PolitiFact)
- Measures model generalization beyond training distribution
- Identifies potential overfitting to stylistic markers

### Phase 6: Model Refinement
- Retraining on challenging FakeNewsNet dataset
- Enhanced robustness against diverse news sources
- Improved generalization capabilities

## Requirements

```
pandas
scikit-learn
transformers
torch
numpy
shap
matplotlib
tqdm
```

## Installation

```bash
pip install pandas scikit-learn transformers torch numpy shap matplotlib tqdm
```

## Dataset Structure

### Required Files

**Initial Training:**
```
data/
├── True.csv      # Real news articles
└── Fake.csv      # Fake news articles
```

**Transfer Learning & Refinement:**
```
testdata/
├── gossipcop_fake.csv
├── gossipcop_real.csv
├── politifact_fake.csv
└── politifact_real.csv
```

### Expected CSV Columns
- `title`: Article headline
- `text`: Article content
- `subject`: Topic category (optional)
- `date`: Publication date (optional)

Alternative column names supported:
- `news_content`: Full article text
- Auto-detection of available columns with intelligent fallback

## Configuration

Key hyperparameters (modifiable at the top of the script):

```python
RANDOM_SEED = 42           # Reproducibility
MAX_LEN = 128              # Maximum token sequence length
BATCH_SIZE = 16            # Training batch size
NUM_EPOCHS = 1             # Training epochs
BERT_MODEL_NAME = 'bert-base-uncased'
```

## Usage

### Basic Execution

```bash
python fake_news_detection.py
```

The script will automatically:
1. Load and preprocess datasets
2. Train the BERT classifier
3. Evaluate performance
4. Generate SHAP explanations
5. Test on alternative datasets (if available)
6. Retrain for improved robustness

### GPU Acceleration

The model automatically detects and utilizes CUDA-enabled GPUs. If CUDA out of memory errors occur, reduce `BATCH_SIZE`:

```python
BATCH_SIZE = 8  # or 4 for limited GPU memory
```

## Output

### Console Output
- Dataset statistics and label distribution
- Training progress with loss metrics
- Validation performance per epoch
- Test set evaluation metrics
- Confusion matrix interpretation
- SHAP value computations

### Visualizations
- Confusion matrix heatmap
- SHAP text plots showing word-level contributions

### Saved Artifacts
```
results/              # Model checkpoints and training logs
logs/                 # TensorBoard-compatible logs
results_fakenewsnet/  # Refined model checkpoints
logs_fakenewsnet/     # Refined model logs
```

## Model Performance Interpretation

### Metrics Explained
- **Accuracy**: Overall correctness (TP + TN) / Total
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Confusion Matrix
```
                    Predicted True | Predicted Fake
Actual True (0):    TN             | FP (Type I Error)
Actual Fake (1):    FN (Type II)   | TP
```

## SHAP Explanations

The system provides word-level explanations for predictions:
- **Red words**: Push prediction toward FAKE (label 1)
- **Blue words**: Push prediction toward TRUE (label 0)
- **Intensity**: Indicates strength of contribution

## Error Handling

The script includes robust error handling for:
- Missing dataset files (falls back to synthetic examples)
- CUDA out of memory errors (with helpful suggestions)
- Missing columns in CSV files (auto-detection)
- Empty or invalid text entries

## Architecture

### Model Pipeline
```
Input Text → BERT Tokenizer → Token IDs + Attention Masks → 
BERT Encoder → [CLS] Token Representation → 
Linear Classifier → Softmax → Binary Prediction (0=True, 1=Fake)
```

### Custom Components
- `NewsDataset`/`FakeNewsDataset`: PyTorch Dataset wrapper for tokenization
- `predict_fn`/`predict_fn_final`: SHAP-compatible prediction functions
- `compute_metrics`: Multi-metric evaluation callback

## Limitations & Considerations

1. **Computational Requirements**: BERT models require significant GPU memory
2. **Training Time**: Full training may take hours depending on dataset size
3. **Context Window**: Limited to 128 tokens (expandable via `MAX_LEN`)
4. **Dataset Bias**: Initial model may overfit to stylistic markers in training data
5. **Language**: Currently optimized for English text only

## Future Improvements

- Multi-language support using multilingual BERT
- Ensemble methods combining multiple models
- Real-time inference API
- Extended context windows for longer articles
- Fine-grained misinformation categorization
- Integration with fact-checking databases


## Acknowledgments

- Hugging Face Transformers library
- SHAP explainability framework
- FakeNewsNet dataset creators
- BERT authors (Devlin et al., 2019)

## Contact

For questions or issues, please open an issue on the repository or contact nvshanmukh28@gmail.com.

---

**Note**: This project is for educational and research purposes. Always verify important information through multiple reliable sources.