import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def calculate_svm_feature_importance(model, X, y, n_repeats=10):
    """
    Calculate feature importance for SVM models using permutation importance.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained SVM model
    X : array-like of shape (n_samples, n_features)
        Input data used for calculating feature importance
    y : array-like of shape (n_samples,)
        Target values
    n_repeats : int, default=10
        Number of times to permute each feature
        
    Returns:
    --------
    feature_importance : array of shape (n_features,)
        Mean importance of each feature
    feature_importance_std : array of shape (n_features,)
        Standard deviation of feature importance
    """
    # Get baseline score
    baseline_score = r2_score(y, model.predict(X))
    
    # Initialize arrays to store importance scores
    n_features = X.shape[1]
    importance_scores = np.zeros((n_repeats, n_features))
    
    # Calculate importance for each feature
    for feature_idx in range(n_features):
        for rep_idx in range(n_repeats):
            # Create a copy of the data
            X_permuted = X.copy()
            
            # Permute the feature
            rng = np.random.RandomState(rep_idx)
            permutation_idx = rng.permutation(len(X))
            X_permuted[:, feature_idx] = X_permuted[permutation_idx, feature_idx]
            
            # Calculate score with permuted feature
            permuted_score = r2_score(y, model.predict(X_permuted))
            
            # Calculate importance as decrease in score
            importance_scores[rep_idx, feature_idx] = baseline_score - permuted_score
    
    # Calculate mean and std of importance scores
    feature_importance = np.mean(importance_scores, axis=0)
    feature_importance_std = np.std(importance_scores, axis=0)
    
    return feature_importance, feature_importance_std

def plot_feature_importance(feature_importance, feature_importance_std, feature_names=None):
    """
    Plot feature importance with error bars.
    
    Parameters:
    -----------
    feature_importance : array-like
        Mean importance scores for each feature
    feature_importance_std : array-like
        Standard deviation of importance scores
    feature_names : list or None
        Names of features. If None, will use feature indices
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(feature_importance))]
    
    # Sort features by importance
    idx = np.argsort(feature_importance)
    importance_sorted = feature_importance[idx]
    std_sorted = feature_importance_std[idx]
    names_sorted = np.array(feature_names)[idx]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance_sorted)), importance_sorted,
             xerr=std_sorted, capsize=5)
    plt.yticks(range(len(importance_sorted)), names_sorted)
    plt.xlabel('Feature Importance (decrease in R² score)')
    plt.title('SVM Feature Importance')
    plt.tight_layout()
    
    return plt.gcf()  

