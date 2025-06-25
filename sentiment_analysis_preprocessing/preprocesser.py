import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer for preprocessing text.

    This Preprocessor performs the following steps:
    1. Removes all non-alphabetic characters.
    2. Converts the text to lowercase.
    3. Tokenizes the text by splitting on whitespace.
    4. Removes English stopwords (while keeping 'not').
    5. Applies the Porter Stemmer to each token.
    6. Joins the tokens back into a single string.
    """
    def __init__(self):
        self.ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        self.stopwords_set = set(all_stopwords)
        
    def fit(self, X, y=None):
        """
        This preprocessor is "stateless" (it doesn't learn any parameters
        from the data), so the fit method does nothing and simply returns self.

        The fit method is included to comply with the scikit-learn API.
        """
        return self

    def transform(self, X: iter) -> list:
        """
        Applies the preprocessing steps to an iterable of text documents.

        Args:
            X: An iterable of strings (e.g., a list of reviews or a pandas Series).

        Returns:
            A list of the processed strings.
        """
        return [self._process(item) for item in X]

    def _process(self, item: str) -> str:
        """
        Processes a single text item by applying the preprocessing steps.
        Args:
            item: A single string to be processed.

        Returns:
            A processed string with all preprocessing steps applied.
        """
        review = re.sub('[^a-zA-Z]', ' ', item)
        review = review.lower()
        review_tokens = review.split()
        
        processed_tokens = [
            self.ps.stem(word)
            for word in review_tokens
            if word not in self.stopwords_set
        ]
        
        return ' '.join(processed_tokens)