# Lib-ml

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

A comprehensive Python library for text preprocessing and feature engineering, specifically designed for sentiment analysis and natural language processing tasks. This library provides a complete pipeline for transforming raw text data into machine learning-ready numerical representations.

> **Note**: This library is designed for educational and research purposes in the context of Release Engineering for Machine Learning Applications (REMLA).
> 
## Local Development Setup

### Prerequisites
- Python 3.6 or higher
- Git
- pip (Python package installer)

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/remla2025-team9/lib-ml.git
   cd lib-ml
   ```

2. **Create and activate a virtual environment** (recommended)
   
   **On Windows:**
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   .venv\Scripts\activate
   ```
   
   **On macOS/Linux:**
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   source .venv/bin/activate
   ```

3. **Upgrade pip** (recommended)
   ```bash
   python -m pip install --upgrade pip
   ```

4. **Install the dependencies**
   ```bash
   # Install dependencies
    pip install -r requirements.txt
   ```



## Installation

### From GitHub Release
```bash
pip install git+https://github.com/remla2025-team9/lib-ml.git
```


##  Quick Start

### Basic Usage

```python
from sentiment_analysis_preprocessing.preprocess import preprocess, prepare, load_data

# Load your data
messages = load_data('data/training_data.tsv', 'Review')

# Preprocess and save the pipeline
transformed_data = preprocess(messages, save=True)

# Use the saved pipeline for new data
new_message = "This restaurant was absolutely amazing!"
features = prepare(new_message)
```


## 📚 API Reference

### Core Functions

#### `preprocess(data, save=True, model_path='preprocessor.joblib', data_path='preprocessed_data.joblib')`
Build and fit the complete preprocessing pipeline on your dataset.

**Parameters:**
- `data` (list): List of text strings to preprocess
- `save` (bool): Whether to save the fitted pipeline and transformed data
- `model_path` (str): Path to save the fitted preprocessor
- `data_path` (str): Path to save the transformed data

**Returns:**
- Sparse matrix of preprocessed features combining TF-IDF vectors and message lengths

#### `prepare(message, model_path='output/preprocessor.joblib')`
Transform a single message using a pre-trained pipeline.

**Parameters:**
- `message` (str): Text message to preprocess
- `model_path` (str): Path to the saved preprocessor

**Returns:**
- Sparse matrix of features for the input message

#### `load_data(path, text_col, sep='\t', quoting=3)`
Load text data from CSV/TSV files.

**Parameters:**
- `path` (str): Path to the data file
- `text_col` (str): Column name containing text data
- `sep` (str): Delimiter character (default: tab)
- `quoting` (int): Quote handling mode (default: 3 for QUOTE_NONE)

**Returns:**
- List of text strings from the specified column
  


## Detailed Examples

### Working with Custom Data

```python
# Load custom dataset
data = [
    "I love this product! Amazing quality.",
    "Terrible experience, would not recommend.",
    "Average product, nothing special.",
    "Outstanding service and fast delivery!"
]

# Preprocess with custom settings
features = preprocess(
    data, 
    save=True, 
    model_path='my_preprocessor.joblib',
    data_path='my_features.joblib'
)

print(f"Feature matrix shape: {features.shape}")
print(f"Feature density: {features.nnz / features.size:.4f}")
```

### Batch Processing Large Datasets

```python
import pandas as pd
from sentiment_analysis_preprocessing.preprocess import load_data, preprocess

# Load large dataset
large_dataset = load_data(
    'data/large_reviews.csv', 
    text_col='review_text',
    sep=',',
    quoting=0  # Standard CSV quoting
)

print(f"Loaded {len(large_dataset)} reviews")

# Process in batches if needed
batch_size = 1000
for i in range(0, len(large_dataset), batch_size):
    batch = large_dataset[i:i+batch_size]
    features = preprocess(
        batch, 
        save=True, 
        model_path=f'preprocessor_batch_{i//batch_size}.joblib'
    )
    print(f"Processed batch {i//batch_size + 1}")
```

### Integration with Machine Learning Models

```python
from sentiment_analysis_preprocessing.preprocess import preprocess, prepare
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load and preprocess training data
messages = load_data('data/training_data.tsv', 'Review')
labels = load_data('data/training_data.tsv', 'Liked')  # Assuming binary labels

features = preprocess(messages, save=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test on new data
new_review = "The food was delicious and service was excellent!"
new_features = prepare(new_review)
prediction = model.predict(new_features)

print(f"Sentiment prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
```

## Data Format

The library expects text data in the following formats:

### TSV/CSV Files
```
Review	Liked
Wow... Loved this place.	1
Crust is not good.	0
Not tasty and the texture was just nasty.	0
```

### Python Lists
```python
texts = [
    "Great restaurant with amazing food!",
    "Poor service and cold food.",
    "Average dining experience."
]
```