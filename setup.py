from setuptools import setup, find_packages

setup(
    name="sentiment_analysis_preprocessing",
    packages=find_packages(),
    install_requires=[
        "nltk>=3.5",
        "scikit-learn>=1.0",
        "joblib>=1.0",
        "numpy>=1.19",
        "pandas>=1.3.3"
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
