"""AktaSense ML Pipeline package.

This package implements the reusable Machine Learning building blocks for the
AktaSense classification system: document embedding (IndoBERT + mean pooling)
followed in later sprints by a Random Forest classifier.

Modules
-------
config
    Centralised configuration values for the pipeline.
embedding
    Reusable functions to load data, load the IndoBERT model, tokenize
    documents and build contextual feature matrices.
"""

__version__ = "0.1.0"