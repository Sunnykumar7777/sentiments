"""
This module builds text features from processed data for spam classification.
It creates TF-IDF features and word count features from the input text data.
"""

import click
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
from pathlib import Path
import logging
from dotenv import find_dotenv, load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump

# nltk.download('punkt')

def count_total_words(text):
    """Count the total number of words in a text string using NLTK word tokenization."""
    return len(word_tokenize(text))

def create_tfidf_features(df, text_column='statement', max_features=5000):
    """
    Create TF-IDF and word count features from text data.
    
    Args:
        df (pd.DataFrame): Input dataframe containing text data
        text_column (str): Name of column containing text data. Defaults to 'v2'
        max_features (int): Maximum number of TF-IDF features to create. Defaults to 4000
        
    Returns:
        tuple: (final_df, vectorizer) where:
            - final_df (pd.DataFrame): DataFrame containing TF-IDF features, word counts, and labels
            - vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer object
   """
    logger = logging.getLogger(__name__)
    logger.info('Creating word count feature')

    df[text_column] = df[text_column].astype(str)
    df['num_words'] = df[text_column].apply(count_total_words)
    
    logger.info('Creating TF-IDF features')
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=max_features,
        lowercase=True,
    )
    bow_matrix = vectorizer.fit_transform(df[text_column])
    
    logger.info('Converting TF-IDF matrix to dataframe')
    bow_matrix_df = pd.DataFrame(bow_matrix.toarray())
    num_w_df = pd.DataFrame(df['num_words'])
    y_df = pd.DataFrame(df['status'])
    
    bow_matrix_df = bow_matrix_df.reset_index(drop=True)
    num_w_df = num_w_df.reset_index(drop=True)
    final_df = pd.concat([bow_matrix_df, num_w_df, y_df], axis=1)
    
    final_df = final_df.rename(str, axis="columns")
    
    return final_df, vectorizer


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def build_features(input_filepath, output_filepath):
    """
    Build features from processed data.
    
    Args:
        input_filepath (str): Path to input CSV file containing processed text data
        output_filepath (str): Path where output feature matrix will be saved
        
    The function:
    1. Loads processed text data
    2. Creates TF-IDF and word count features
    3. Saves feature matrix and TF-IDF vectorizer
    """
    logger = logging.getLogger(__name__)
    
    logger.info('Loading processed data')
    df = pd.read_csv(input_filepath)
    
    # Create features
    final_df, vectorizer = create_tfidf_features(df)

    
    final_df.to_csv(output_filepath, index=False)
    logger.info(f'Features built and saved to {output_filepath}')

    output_path = Path(output_filepath)
    vectorizer_path = output_path.parent / 'tfidf_vectorizer.joblib'
    dump(vectorizer, vectorizer_path)
    logger.info(f'Vectorizer saved to {vectorizer_path}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    load_dotenv(find_dotenv())
    build_features()