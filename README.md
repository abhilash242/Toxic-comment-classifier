Toxic Comment Classification
Table of Contents
Functionalities
Dataset Overview
Data Preprocessing and EDA
Model Fitting
Results
Functionalities
Detects Types of Toxicity:

Toxic
Severe Toxic
Obscene
Threat
Insult
Identity Hate
Text Preprocessing:

Removes punctuations and special characters.
Applies stemming and/or lemmatization.
Filters out very short comments.
Vectorization:

Converts text to numerical format using TFIDFVectorizer for efficient model training.
Handles Imbalanced Classes:

Resampling techniques for better representation of minority classes.
Implements robust metrics to evaluate model performance.
Model Comparisons:

Benchmarks multiple models including Naive Bayes, Logistic Regression, and Linear SVC.
Hyperparameter Tuning:

Uses grid search and manual tuning to optimize model performance.
Ensembling:

Combines multiple models to improve prediction accuracy.
Evaluation Metrics:

Uses F1 Score, Recall, and Hamming Loss for comprehensive evaluation.
Feature Analysis:

Identifies top and bottom contributing words for interpretability.
Optimal Model:

Recommends the best-performing model for deployment.
Dataset Overview
Text comments labeled with types of toxicity.
Consists of training and testing data.
Imbalanced class distribution requires careful handling.
Data Preprocessing and EDA
Custom tokenization, stemming, and filtering.
Vectorization using TFIDF.
Addressed class imbalance with appropriate techniques.
Model Fitting
Compared models without tuning to establish baselines.
Tuned hyperparameters for better performance.
Used ensembling for improved results.
Results
Optimal model: Linear SVC based on performance, speed, and interpretability.
Achieved significant improvements using ensembling and advanced evaluation techniques.
