# -*- coding: utf-8 -*-
import click
import mmap
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath , output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    ## Read data
    # Read from csv
    input_referencepath = f"{project_dir}/references/dataset_master.csv"
    data = pd.read_csv(input_referencepath, sep=';', decimal=',')

    # Gather tools
    tools = list(data['tool'].unique())
    n = 5

    ## Gather training and testing data
    # Process data
    for toolnr in tqdm(tools):
        bitdicts = []

        print(f"Gathering files from tool {toolnr}")
        relevant_filenames = data.loc[data['tool'] == toolnr, 'stegoPictureName'].tolist()
        for filename in tqdm(relevant_filenames):
            bitdict = _load_image_as_bitdict(f"{input_filepath}/{filename}", n)
            bitdict['tool'] = toolnr
            bitdicts.append(bitdict)

        # Transform data
        df = pd.DataFrame(bitdicts)
        df.to_csv(f'{output_filepath}/{toolnr}_full_{n*8}bits.csv')

def _load_image_as_bitdict(path: str, max_offset=1000) -> dict:
    filebits = {}
    with open(path, 'rb') as f:
        with mmap.mmap(f.fileno(), length=max_offset, access=mmap.ACCESS_READ) as mm:
            for i in range(len(mm) * 8):
                byte_index = i // 8
                bit_index = 7 - (i % 8)
                bt = (mm[byte_index] >> bit_index) & 1
                filebits[i] = bt

    return filebits

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    input_dir = '/Volumes/New Volume/Public_Set_Stego_Pictures'
    output_dir = f"{project_dir}/data/processed/bitstreams"

    main()
