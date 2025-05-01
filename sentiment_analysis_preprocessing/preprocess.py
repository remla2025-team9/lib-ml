import re
import os
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_union
import numpy as np

nltk.download('stopwords')

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ps = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
        self.stopwords.discard('not')  # keep "not" for sentiment

    def _clean_text(self, text):
        text = re.sub('[^a-zA-Z]', ' ', text).lower()
        tokens = text.split()
        tokens = [self.ps.stem(w) for w in tokens if w not in self.stopwords]
        return ' '.join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._clean_text(text) for text in X]

def extract_message_len(data):
    return np.array([len(text) for text in data]).reshape(-1, 1)

def preprocess(data, save=True, model_path='preprocessor.joblib', data_path='preprocessed_data.joblib'):
    """
    Build pipeline, fit on dataset (list of texts), and save transformer and output.
    """
    pipeline = make_union(
        make_pipeline(
            TextCleaner(),
            CountVectorizer(),
            TfidfTransformer()
        ),
        FunctionTransformer(extract_message_len, validate=False)
    )

    transformed = pipeline.fit_transform(data)

    if save:
        os.makedirs('output', exist_ok=True)
        joblib.dump(pipeline, f'output/{model_path}')
        joblib.dump(transformed, f'output/{data_path}')
        print(f"Preprocessor and transformed data saved to 'output/'.")

    return transformed

def prepare(message, model_path='output/preprocessor.joblib'):
    """
    Load pipeline and transform a single message.
    """
    pipeline = joblib.load(model_path)
    return pipeline.transform([message])


