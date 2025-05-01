import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Initialize the PorterStemmer
ps = PorterStemmer()

# Get the stopwords for English and remove 'not'
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

def preprocess_text(text: str) -> str:
    """
    Preprocesses a single review text by:
    1. Removing non-alphabetic characters
    2. Converting to lowercase
    3. Tokenizing the text
    4. Removing stopwords (except 'not')
    5. Stemming the words

    Args:
        text (str): The raw text to be processed.

    Returns:
        str: The processed text.
    """
    review = re.sub('[^a-zA-Z]', ' ', text)
    
    review = review.lower()

    review = review.split()

    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]

    processed_review = ' '.join(review)
    
    return processed_review
