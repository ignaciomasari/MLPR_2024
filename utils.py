# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:47:03 2020

@author: David
"""

from osgeo import gdal

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pprint
import pathlib


def load_image(path_str):
    """Load a GeoTiff image into a numpy array."""
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    ds = gdal.Open(path_str)

    raster = np.ones((ds.RasterYSize,ds.RasterXSize,ds.RasterCount), dtype=int)
    for i in range(ds.RasterCount):
        raster[:,:,i] = ds.GetRasterBand(i+1).ReadAsArray()
        
    if np.ndim(raster) == 3 and np.shape(raster)[2] == 1:
        return np.squeeze(raster)
    else:
        return raster
    

def save_image(path_str, image):
    """Write an image to file in GeoTiff format."""   
    path = pathlib.Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True) 
    
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    
    n_dim = np.ndim(image)
    n_rows = np.shape(image)[0]
    n_cols = np.shape(image)[1]
    if n_dim == 2:
        n_bands = 1
    elif n_dim == 3:
        n_bands = np.shape(image)[2]
    else:
        raise ValueError("Expected array dimension to be 2 or 3. Received {}.".format(n_dim))
    
    outds = driver.Create(path_str, xsize = n_cols, ysize=n_rows, bands=n_bands, eType = gdal.GDT_Byte)
    outds.SetGeoTransform((0.0,1.0,0.0,0.0,0.0,1.0))
    outds.SetProjection('')
    for i in range(n_bands):
        outband = outds.GetRasterBand(i+1)
        if (n_bands==1):
            outband.WriteArray(image[:,:])
        else:
            outband.WriteArray(image[:,:,i])
        outband.SetNoDataValue(np.nan)
        outband.FlushCache()
        
        
def normalize_datacube_0_1(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array = np.copy(array.astype(np.float64))
    n_dim = np.ndim(array)
    if n_dim == 3:
        bands = np.shape(array)[2]
        for b in range(0, bands):
            band = array[:,:,b]
            band_min, band_max = band.min(), band.max()
            array[:,:,b] = ((band - band_min)/(band_max - band_min))
        return array
    elif n_dim == 2:
        array_min, array_max = array.min(), array.max()
        return ((array - array_min)/(array_max - array_min))
    else:
        raise ValueError("Expected array dimension to be 2 or 3. Received {}.".format(n_dim))
        
        
def stretch_classification_map(c_map):
    """Stretch a classification map between 0 and 255 for visualization."""
    c_map = normalize_datacube_0_1(c_map) * 255
    return c_map.astype("uint8")
    
    
def export_to_xlsx(path_str, data):
    """Export some data to an Excel file."""
    path = pathlib.Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True) 
    df = pd.DataFrame(data)
    writer = pd.ExcelWriter(path_str, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    

def save_dict(path_str, data):
    """Export a dictionary to file with nice formatting."""
    path = pathlib.Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True) 
    pp = pprint.PrettyPrinter(indent=4)
    output_string = pp.pformat(data)
    with open(path_str, 'w') as file:
        file.write(output_string)


def show_image(array, title=None):
    """Show an image."""
    if np.ndim(array) < 2 or np.ndim(array) > 3:
        raise ValueError("The image must be either 2D or 3D.")
        
    if np.ndim(array) == 3:
        n_bands = np.shape(array)[2]
        if n_bands == 1:
            array = np.squeeze(array)
        elif n_bands == 2:
            array = np.dstack([array, array[:,:,1]])
        elif n_bands > 3:
            array = array[:,:,0:3]
    
    plt.figure()     
    plt.title(title)
    plt.imshow(array)