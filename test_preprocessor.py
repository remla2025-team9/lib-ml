from sentiment_analysis_preprocessing.preprocesser import Preprocessor

def test_preprocessor():
    preprocessor = Preprocessor()
    
    # Test cases
    test_cases = [
        "This is great!",
        "I don't like this product.",
        "The restaurant was AMAZING!!",
        "Very bad experience :(",
    ]
    
    processed = preprocessor.transform(test_cases)
    
    # Print original and processed versions
    for original, processed_text in zip(test_cases, processed):
        print(f"Original: {original}")
        print(f"Processed: {processed_text}")
        print("-" * 40)

if __name__ == "__main__":
    test_preprocessor()