"""
This module implements model training and evaluation for spam classification.
It loads features, trains Random Forest and Gradient Boosting classifiers, 
and evaluates their performance on test data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import logging
from pathlib import Path
import joblib
import click
from dotenv import find_dotenv, load_dotenv
import yaml
import json


def load_params(params_path):
    """Load model parameters from YAML config file."""
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_features(features_filepath):
    """Load feature matrix from CSV file."""
    logger = logging.getLogger(__name__)
    logger.info('Loading features')
    return pd.read_csv(features_filepath)


def train_model_gb(X_train, y_train, n_estimators, random_state, max_depth, learning_rate):
    """
    Train Gradient Boosting classifier.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        n_estimators (int): Number of boosting stages
        random_state (int): Random seed for reproducibility
        max_depth (int): Maximum depth of trees
        
    Returns:
        GradientBoostingClassifier: Trained Gradient Boosting model
    """
    logger = logging.getLogger(__name__)
    logger.info('Training GradientBoosting model')
    
    gb_classifier = GradientBoostingClassifier(
        n_estimators=n_estimators, 
        random_state=random_state,
        max_depth=max_depth,
        learning_rate=learning_rate
    )
    
    gb_classifier.fit(X_train, y_train)
    return gb_classifier


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test data."""
    logger = logging.getLogger(__name__)
    logger.info('Evaluating model')
    
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    logger.info(f'Model accuracy: {accuracy:.2f}')
    
    return accuracy



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
# @click.argument('output_filepath2', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Main function to train and evaluate models.
    
    Args:
        input_filepath (str): Path to input features file
        output_filepath (str): Path to save Random Forest model
        output_filepath2 (str): Path to save Gradient Boosting model
        
    The function:
    1. Loads model parameters and feature data
    2. Splits data into train/test sets
    3. Trains Random Forest and optionally Gradient Boosting models
    4. Evaluates model performance
    5. Saves trained models
    """
    logger = logging.getLogger(__name__)
    
    params_path = Path(__file__).resolve().parents[2] / 'params.yaml'
    params = load_params(params_path)
    
    final_df = load_features(input_filepath)
    
    target_column = params['data']['target_column']
    X = final_df.drop([target_column], axis=1)
    y = final_df[target_column]
    
    logger.info('Splitting data into train and test sets')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['data']['test_size'], 
        random_state=params['data']['random_state']
    )
    
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data_path = Path(input_filepath).parent / 'test_data.csv'
    logger.info(f'Saving test data to {test_data_path}')
    test_data.to_csv(test_data_path, index=False)
    
    model = train_model_gb(
        X_train, 
        y_train, 
        n_estimators=params['model']['gradient_boosting']['n_estimators'],
        random_state=params['model']['gradient_boosting']['random_state'],
        max_depth=params['model']['gradient_boosting']['max_depth'],
        learning_rate=params['model']['gradient_boosting']['learning_rate']
    )
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Save the model
    logger.info(f'Saving model to {output_filepath}')
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())

    main()