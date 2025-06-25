# Lib-ml

A Python library for text preprocessing and feature engineering, designed for sentiment analysis tasks.

## Development Setup

1. Clone the repository
   ```bash
   git clone https://github.com/remla2025-team9/lib-ml.git
   cd lib-ml
   ```

2. Create and activate virtual environment
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Using the Preprocessor Class

Install the library in your project:
```bash
# Install specific version (e.g. v1.0.0)
pip install git+https://github.com/remla2025-team9/lib-ml.git@v1.0.0
```

Import and use the Preprocessor:
```python
from sentiment_analysis_preprocessing.preprocesser import Preprocessor

# Create preprocessor instance
preprocessor = Preprocessor()

# Preprocess training data
training_data = ["This is great!", "This is terrible."]
processed_texts = preprocessor.transform(training_data)

# Preprocess new data
new_texts = ["Amazing product!"]
new_processed = preprocessor.transform(new_texts)
```

The Preprocessor performs:
1. Removes non-alphabetic characters
2. Converts to lowercase
3. Tokenizes by whitespace
4. Removes English stopwords (keeps 'not')
5. Applies Porter Stemming
6. Joins tokens back into string

Note that the Preprocessor is stateless, meaning it does not learn any parameters from the data. Therefore, calling `fit()` or `fit_transform()` is optional and will simply return the preprocessor instance itself or the transformed data,

