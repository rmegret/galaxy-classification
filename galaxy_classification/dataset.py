from pathlib import Path
import os
from shutil import copyfile
from typing import Union

from loguru import logger
from tqdm import tqdm
import typer

from galaxy_classification.config import SMOG_DATA_DIR, RGB_DATA_DIR
import pandas as pd
import re
from galaxy_classification.utils import move_columns_to_front

from PIL import Image
import numpy as np

from .cleaning import refine_center


app = typer.Typer()


def load_dataframe(data: Union[Path, pd.DataFrame], **read_csv_kwargs) -> pd.DataFrame:
    """
    Load a DataFrame from a Path or return the DataFrame if already provided.
    """
    if isinstance(data, Path):
        return pd.read_csv(data,**read_csv_kwargs)
    elif isinstance(data, pd.DataFrame):
        return data
    else:
        raise TypeError("Expected a Path or a pandas DataFrame.")


@app.command()
def create_smog_catalog(
    smog_data_dir: Path = SMOG_DATA_DIR,
    catalog_path: Path = SMOG_DATA_DIR / "smog_catalog.csv",
):
    """
    Create a catalog of SMOG data with galaxy labels.
    """

    logger.info(f"# Scanning SMOG data to generate catalog...")

    ## Scan field folders Galaxias/10450_0145	No_Galaxias/10450_0145

    logger.info(f"Scanning SMOG field folders in {smog_data_dir}...")

    pattern_5_4 = re.compile(r'\d{5}_\d{4}')
    def is5_4(name):
        return bool(pattern_5_4.fullmatch(name))

    items = []

    # Process Galaxias folder
    folder_path = SMOG_DATA_DIR / 'Galaxias'
    folders = [f.name for f in folder_path.iterdir() if f.is_dir() and is5_4(f.name)]
    items.extend([dict(folder=f'Galaxias/{f}', galaxy=1) for f in folders])

    # Process No_Galaxias folder
    folder_path = SMOG_DATA_DIR / 'No_Galaxias'
    folders = [f.name for f in folder_path.iterdir() if f.is_dir() and is5_4(f.name)]
    items.extend([dict(folder=f'No_Galaxias/{f}', galaxy=0) for f in folders])

    # Create DataFrame and save to CSV
    df_topdirs = pd.DataFrame(items, columns=['folder', 'galaxy'])
    
    #logger.info(f"Found {df_topdirs.shape[0]} unique top-level directories:\n{df_topdirs}")
    logger.info(f"  Found {df_topdirs.shape[0]} unique top-level directories")
    logger.info(f"  {df_topdirs.galaxy.sum()} galaxy, {(1-df_topdirs.galaxy).sum()} no-galaxy")


    ## Scan subdirs by imagetype Galaxias/10450_0145/Blue_10450_0145
    logger.info(f"Scanning SMOG imagetype subfolders...")
    
    pattern_color = re.compile(r'^[A-Za-z0-9]+')
    def extract_imagetype(name):
        match = pattern_color.match(name)
        return match.group(0) if match else None

    items=[]
    for _,item in df_topdirs.iterrows():
        folder = item['folder']
        galaxy = item['galaxy']
        folder_path = SMOG_DATA_DIR / folder
        subfolders = [f.name for f in folder_path.iterdir() if f.is_dir()]

        items.extend( [dict(folder=folder, subfolder=f, imagetype=extract_imagetype(f), galaxy=galaxy) for f in subfolders] )

    df_imagedirs = pd.DataFrame(items, columns=['folder', 'subfolder', 'imagetype','galaxy'])
    df_imagedirs

    #logger.info(f"Found {df_imagedirs.shape[0]} subfolders:\n{df_imagedirs}")
    logger.info(f"  Found {df_imagedirs.shape[0]} subfolders")
    logger.info(f"  {df_imagedirs.galaxy.sum()} galaxy, {(1-df_imagedirs.galaxy).sum()} no-galaxy")


    ## Scan all images Galaxias/10450_0145/RGB_10450_0145/104.7263830,2.0560720_10450_0145_RGB-composite.jpeg
    ## And extract metadata
    logger.info(f"Scanning SMOG image files and extracting their metadata...")
    def parse_filename(filename):
        match = re.match(r'^(-?\d+\.\d+),(-?\d+\.\d+)_([0-9]+_[0-9]+)_(.*)\.([\w]+)$', filename)
        if match:
            return dict(ra=match.group(1), dec=match.group(2), field=match.group(3), type=match.group(4), suffix=match.group(5))
        return dict(ra=None, dec=None, field=None, type=None, suffix=None)

    items=[]
    for _,item in df_imagedirs.iterrows():
        D = item.to_dict()
        folder = D['folder']
        subfolder = D['subfolder']
        folder_path = SMOG_DATA_DIR / folder / subfolder
        imagefiles = [f.name for f in folder_path.iterdir() if f.is_file()]
        items.extend( [dict(**D, file=f, **parse_filename(f)) for f in imagefiles] )

    df_images = pd.DataFrame(items, columns=['folder', 'subfolder', 'file', 'imagetype','galaxy', 'ra', 'dec', 'field', 'type', 'suffix'])	
    df_images

    df_images['id_str'] = df_images.apply(lambda x: f"{x.field} {x.ra},{x.dec}" if x.ra is not None else None, axis=1)
    df_images['file_loc'] = df_images.apply(lambda x: f"{x.folder}/{x.subfolder}/{x.file}", axis=1)
    #df_images['source_id'] = df_images['id_str'].astype('category').cat.codes
    df_images = move_columns_to_front(df_images, ['file_loc', 'galaxy', 'id_str', 'imagetype', 'field','ra','dec', 'file'])
    df_images#.sort_values('id_str').head(10)

    #logger.info(f"Found {df_images.shape[0]} images:\n{df_images}")
    logger.info(f"  Found {df_images.shape[0]} images")
    logger.info(f"  {df_images.galaxy.sum()} galaxy, {(1-df_images.galaxy).sum()} no-galaxy")
    

    numna = df_images.isna().sum(axis=0).sum()

    logger.info(f"Sanity check: {numna} NA values")

    df_images.to_csv(catalog_path, index=False)

    logger.info(f"SMOG catalog saved to {catalog_path}")
    
    return df_images


@app.command()
def create_rgb_catalog(
    smog_catalog: Union[Path, pd.DataFrame] = SMOG_DATA_DIR / "smog_catalog.csv",
    rgb_catalog_path: Path = SMOG_DATA_DIR / "rgb_catalog.csv",
):
    """
    Create an RGB dataset based on the SMOG catalog.
    """
    logger.info(f"Loading SMOG catalog {smog_catalog}...")
    df_images = load_dataframe(smog_catalog)
    
    logger.info("Keeping only RGB images...")
    catalog = df_images[df_images.imagetype=='RGB'].copy()

    logger.info("Adapting file_loc to new hierarchy...")
    # Change file_locfor new organization of dataset
    catalog['smog_file_loc'] = catalog['file_loc']
    catalog['file_loc'] = catalog.apply(lambda x: f"{'galaxy' if x.galaxy else 'no_galaxy'}/{x.file}", axis=1)
    
    catalog.reset_index(inplace=True, drop=True) # Get a fresh index for the dataset
    catalog['source_id'] = range(catalog.shape[0])

    catalog = move_columns_to_front(catalog, ['file_loc', 'galaxy', 'id_str', 'source_id', 'file'])

    logger.info(f"Saving to {rgb_catalog_path}...")
    catalog.to_csv(rgb_catalog_path, index=False)
    return catalog


def clean_rgb_catalog(
    rgb_catalog: Union[Path, pd.DataFrame] = RGB_DATA_DIR / "catalog.csv",
    clean_catalog_path: Path = RGB_DATA_DIR / "clean_catalog.csv",
    rgb_data_dir: Path = RGB_DATA_DIR,
) -> pd.DataFrame:
    """
    Filter out bad images from rgb catalog
    """
    logger.info("Cleaning the catalog...")
    rgb_catalog = load_dataframe(rgb_catalog)

    # Process the catalog and refine the center
    for k in tqdm(rgb_catalog.index):
        galaxy = rgb_catalog.loc[k]
        inpath = rgb_data_dir / galaxy['file_loc']
        im = Image.open(inpath)
        im = im.resize((100,100))
        im = im.resize((256,256))

        img = np.array(im).sum(axis=2)
        #img = np.array(im)[:,:,0]
        result = refine_center(img)

        if (result is None):
            center = (None, None)
        else:
            center = result['center']
        rgb_catalog.loc[k, 'center_x'] = center[0]
        rgb_catalog.loc[k, 'center_y'] = center[1]

    clean_catalog = rgb_catalog.drop(rgb_catalog.loc[pd.isna(rgb_catalog.center_x)].index, axis=0)

    if (clean_catalog_path):
        clean_catalog.to_csv(clean_catalog_path, index=False)
    return clean_catalog


@app.command()
def create_rgb_dataset_on_disk(
    rgb_catalog: Union[Path, pd.DataFrame] = SMOG_DATA_DIR / "rgb_catalog.csv",
    rgb_dir: Path = RGB_DATA_DIR,
    smog_data_dir: Path = SMOG_DATA_DIR,
):
    df = load_dataframe(rgb_catalog)

    # Ensure the directories exist
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(f"{rgb_dir}/galaxy", exist_ok=True)
    os.makedirs(f"{rgb_dir}/no_galaxy", exist_ok=True)

    # Do the copying
    for _,item in tqdm(df.iterrows()):
        D = item.to_dict()
        smog_file_loc = D['smog_file_loc']
        file_loc = D['file_loc']
        inpath = smog_data_dir / smog_file_loc
        outpath = rgb_dir / file_loc
        copyfile(inpath, outpath)

    df.to_csv(rgb_dir / 'catalog.csv', index=False)

@app.command()
def create_rgb_thumbnails_on_disk(
    rgb_catalog: Union[Path, pd.DataFrame] = RGB_DATA_DIR / "catalog.csv",
    rgb_dir: Path = RGB_DATA_DIR
):
    df = load_dataframe(rgb_catalog)

    # Ensure the directories exist
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(f"{rgb_dir}/galaxy", exist_ok=True)
    os.makedirs(f"{rgb_dir}/no_galaxy", exist_ok=True)

    # Do the copying
    for key,item in tqdm(df.iterrows()):
        D = item.to_dict()
        file_loc = D['file_loc']
        inpath = rgb_dir / file_loc
        outpath = rgb_dir / (file_loc+'_thumb.jpg')

        im = Image.open(inpath)
        im = im.crop(box=(128-32,128-32,128+32,128+32))
        im = im.resize((32,32))
        pixels = np.array(im)
        pixels[:,:,:3] = (pixels.astype('float32')/pixels[:,:,:3].max()*255.0).astype('uint8')
        im = Image.fromarray(pixels)
        im.save(outpath)
        df.loc[key,'thumb'] = file_loc+'_thumb.jpg'

    df.to_csv(rgb_dir / 'catalog_thumb.csv', index=False)

if __name__ == "__main__":
    app()
