"""
This module handles model evaluation by loading a trained model,
making predictions on test data, and calculating accuracy scores.
"""

import pandas as pd
import logging
from pathlib import Path
import joblib
import click
from dotenv import find_dotenv, load_dotenv
import yaml



def load_params(params_path):
    """Load model parameters from YAML config file."""
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_model(model_path):
    """Load trained model from disk."""
    return joblib.load(model_path)

def make_predictions(model, X_test):
    """Generate predictions using trained model."""
    return model.predict(X_test)

def evaluate_predictions(y_true, y_pred):
    """Calculate accuracy score for predictions."""
    accuracy = (y_true == y_pred).mean()
    return accuracy

@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('test_data_path', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(model_path, test_data_path, output_filepath):
    """
    Main function to evaluate model performance.
    
    Args:
        model_path (str): Path to saved model file
        test_data_path (str): Path to test data CSV
        output_filepath (str): Path to save accuracy results
        
    The function:
    1. Loads model parameters and test data
    2. Loads trained model
    3. Makes predictions on test data
    4. Calculates and saves accuracy score
    """
    logger = logging.getLogger(__name__)
    
    params_path = Path(__file__).resolve().parents[2] / 'params.yaml'
    params = load_params(params_path)
    
    logger.info(f'Loading test data from {test_data_path}')
    test_df = pd.read_csv(test_data_path)
    
    target_column = params['data']['target_column']
    X_test = test_df.drop([target_column], axis=1)
    y_test = test_df[target_column]
    
    logger.info(f'Loading model from {model_path}')
    model = load_model(model_path)
    
    logger.info('Making predictions')
    predictions = make_predictions(model, X_test)
    
    accuracy = evaluate_predictions(y_test, predictions)
    logger.info(f'Model accuracy on test set: {accuracy:.4f}')
    
    logger.info(f'Saving accuracy score to {output_filepath}')
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_filepath, 'w') as f:
        f.write(f'Model Accuracy GB: {accuracy:.4f}')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())

    main()