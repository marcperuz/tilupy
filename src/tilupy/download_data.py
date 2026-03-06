# -*- coding: utf-8 -*-

import requests
import zipfile
import io
import os
import pandas as pd


def import_frankslide_dem(folder_out: str = None, file_out: str = None) -> str:
    """Import frankslide topography.

    Parameters
    ----------
    folder_out : str, optional
        Path to the folder output. If None the current folder will be choosed. By default None.
    file_out : str, optional
        Name of the file. If None choose "Frankslide_topography.asc". By default None.

    Returns
    -------
    str
        Path to the saved file.
    """
    if folder_out is None:
        folder_out = "."
    if file_out is None:
        file_out = "Frankslide_topography.asc"

    file_save = os.path.join(folder_out, file_out)

    url = (
        "https://raw.githubusercontent.com/marcperuz/tilupy/main/data/"
        + "frankslide/rasters/Frankslide_topography.asc"
    )
    r = requests.get(url)
    open(file_save, "w").write(r.text)

    return file_save


def import_frankslide_pile(folder_out: str = None, file_out: str = None) -> str:
    """Import frankslide pile.

    Parameters
    ----------
    folder_out : str, optional
        Path to the folder output. If None the current folder will be choosed. By default None.
    file_out : str, optional
        Name of the file. If None choose "Frankslide_pile.asc". By default None.

    Returns
    -------
    str
        Path to the saved file.
    """

    if folder_out is None:
        folder_out = "."
    if file_out is None:
        file_out = "Frankslide_pile.asc"

    file_save = os.path.join(folder_out, file_out)

    url = (
        "https://raw.githubusercontent.com/marcperuz/tilupy/main/data/"
        + "frankslide/rasters/Frankslide_pile.asc"
    )
    r = requests.get(url)
    open(file_save, "w").write(r.text)

    return file_save


def import_shaltop_frankslide(folder_out: str = "./shaltop_frankslide"):
    """Import shaltop results for the Frankslide.

    Parameters
    ----------
    folder_out : str, optional
        Folder where data will be saved. By default "./shaltop_frankslide".

    Returns
    -------
    None

    """
    url = (
        "https://raw.githubusercontent.com/marcperuz/tilupy/"
        + "main/data/shaltop/shaltop_frankslide.zip"
    )
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(folder_out)


def import_shaltop_mus_calibrated(sep=";"):
    """Import database of friction coefficients calibrated with Shaltop with the Coulomb rheology"""
    
    # Load data
    url = "https://zenodo.org/records/18791119/files/mus_calibrated.csv"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    contenu = io.StringIO(response.text)
    df = pd.read_csv(contenu, sep=sep)

    # Get publication date
    params = {
        "sort": "mostrecent",
        "page": 1,
        "size": 1,  # On veut juste la plus récente
        "q": "conceptrecid:18791118"
    }
    response = requests.get("https://zenodo.org/api/records", params=params)
    if response.status_code != 200:
        raise Exception(f"Request error : {response.status_code}, {response.text}")
    data = response.json()
    if data["hits"]["hits"]:
        latest = data["hits"]["hits"][0]
        publication_date =  latest["metadata"]["publication_date"]
    else:
        raise Exception("conceptrecid 18791118 not found.")

    return df, publication_date
