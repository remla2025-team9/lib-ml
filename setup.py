from setuptools import setup, find_packages

setup(
    name="sentiment_analysis_preprocessing",
    version="0.1.0",  
    packages=find_packages(),  # Automatically find all packages, including the current package
    install_requires=[  # List any dependencies here
        "nltk", "scikit-learn", "tensorflow",
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
