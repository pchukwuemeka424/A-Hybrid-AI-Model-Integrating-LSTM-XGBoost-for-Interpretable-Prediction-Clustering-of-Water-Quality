
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Explainable AI for Water Quality Analysis
This module provides comprehensive model explainability using SHAP and LIME
for water quality prediction using Random Forest classifier.
"""

# Install required packages (uncomment if running locally)
# !pip install shap lime matplotlib seaborn pandas numpy scikit-learn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import classification_report, accuracy_score
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def load_and_preprocess_data(file_path="Water Parameters.csv"):
    """
    Load and preprocess water quality data
    
    Returns:
        X: Features DataFrame
        y: Target variable (discretized pH levels)
        y_continuous: Original continuous pH values
        binner: KBinsDiscretizer instance
    """
    # Load data
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully with {df.shape[0]} samples and {df.shape[1]} features")
    
    # Prepare features and target
    X = df.drop(columns=["FID", "Lat", "long", "Town", "pH"])
    y_continuous = df["pH"]
    
    # Discretize pH into categories
    binner = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    y = binner.fit_transform(y_continuous.values.reshape(-1, 1)).ravel()
    
    print("\nTarget categories created:")
    bin_edges = binner.bin_edges_[0]
    for i, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        print(f"Class {i} (pH): [{lower:.2f}, {upper:.2f}]")
    
    return X, y, y_continuous, binner

def train_model(X_train, y_train):
    """
    Train Random Forest classifier
    
    Returns:
        trained_model: Trained Random Forest classifier
    """
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Model training complete!")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Returns:
        y_pred: Predicted values
        report: Classification report
        accuracy: Accuracy score
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)
    
    return y_pred, report, accuracy

def shap_analysis(model, X_train, X_test):
    """
    Perform SHAP explainability analysis
    
    Returns:
        explainer: SHAP TreeExplainer instance
        shap_values: SHAP values
    """
    print("\n=== SHAP Analysis ===")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.title("SHAP Feature Importance (Summary)", fontsize=14)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary plot with beeswarm
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test)
    plt.tight_layout()
    plt.savefig("shap_beeswarm_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return explainer, shap_values

def lime_analysis(model, X_train, X_test, feature_names, class_names):
    """
    Perform LIME explainability analysis
    
    Returns:
        explainer_lime: LIME TabularExplainer instance
    """
    print("\n=== LIME Analysis ===")
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        discretize_continuous=True
    )
    
    # Explain first 3 instances
    for i in range(3):
        print(f"\nExplaining instance {i}:")
        exp = explainer_lime.explain_instance(
            X_test.values[i], 
            model.predict_proba,
            num_features=5
        )
        
        # Display explanation in notebook
        exp.show_in_notebook(show_table=True)
        
        # Save as HTML
        exp.save_to_file(f"lime_explanation_instance_{i}.html")
    
    return explainer_lime

def main():
    """
    Main execution function
    """
    print("=== Enhanced Explainable AI for Water Quality ===")
    
    # Load and preprocess data
    X, y, y_continuous, binner = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # Stratify to maintain class distribution
    )
    print(f"\nData split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    y_pred, report, accuracy = evaluate_model(model, X_test, y_test)
    
    # SHAP Analysis
    explainer, shap_values = shap_analysis(model, X_train, X_test)
    
    # LIME Analysis
    class_names = ['Acidic', 'Neutral', 'Alkaline']
    explainer_lime = lime_analysis(model, X_train, X_test, X.columns.tolist(), class_names)
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- shap_summary_plot.png")
    print("- shap_beeswarm_plot.png")
    print("- lime_explanation_instance_0.html")
    print("- lime_explanation_instance_1.html")
    print("- lime_explanation_instance_2.html")

if __name__ == "__main__":
    main()
