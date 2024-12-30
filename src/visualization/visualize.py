import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import yaml
import click
import re
from collections import Counter
from dotenv import find_dotenv, load_dotenv
import dvclive
from wordcloud import WordCloud


def load_params(params_path):
   """Load parameters from YAML config file."""
   with open(params_path, 'r') as f:
       params = yaml.safe_load(f)
   return params

def plot_data_balance(df, output_path, target_column):
    value_counts = df[target_column].value_counts()

    plt.figure(figsize=(8, 4))
    value_counts.plot(kind='bar')
    plt.title('Types of Status')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.xticks(rotation=0)

    for i, count in enumerate(value_counts):
        plt.text(i, count, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(str(output_path / 'data_balance.png'))
    plt.close()


def plot_wordcloud(df, output_path):
    text = ' '.join(df['statement'].fillna('').astype(str))

    wordcloud = WordCloud(width=800, height=400,
                        background_color='white',
                        min_font_size=10,
                        max_words=100).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(str(output_path / 'Word_cloud.png'))
    plt.close()


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Main function to generate visualizations.
    
    Args:
        input_filepath (str): Path to input data CSV
        output_filepath (str): Directory to save visualization plots
        
    The function:
    1. Loads data and parameters
    2. Creates plot of class distribution
    3. Creates plots of top words for spam and non-spam messages
    """
    logger = logging.getLogger(__name__)
    
    params_path = Path(__file__).resolve().parents[2] / 'params.yaml'
    params = load_params(params_path)

    logger.info(f'Loading data from {input_filepath}')
    df = pd.read_csv(input_filepath)

    output_path = Path(output_filepath)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info('Generating data balance plot')
    plot_data_balance(df, output_path, params['data']['target_column'])

    logger.info('Generating Word Cloud plot')
    plot_wordcloud(df, output_path)


if __name__ == '__main__':
   log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   logging.basicConfig(level=logging.INFO, format=log_fmt)

   project_dir = Path(__file__).resolve().parents[2]
   load_dotenv(find_dotenv())
   main()   