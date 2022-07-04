import os

# folder path or file path constants that will be used in this project

# Root Directory
D_DIR = "D:"
F_DIR = "F:"


# Folder inside D:\\wildfire-sumatera-dataset
WILDFIRE_SUMATERA_DATASET_FOLDER_PATH = f"{F_DIR}\\wildfire-sumatera-dataset"


# Folders and metadatas inside D:\\wildfire-sumatera-dataset
WILDFIRE_SUMATERA_GEOTIFF_FOLDER_PATH        = f"{WILDFIRE_SUMATERA_DATASET_FOLDER_PATH}\\wildfire-sumatera-geotiff"
WILDFIRE_SUMATERA_JPEG_FOLDER_PATH           = f"{WILDFIRE_SUMATERA_DATASET_FOLDER_PATH}\\wildfire-sumatera-jpeg"
# Files (.csv) and metadatas inside D:\\wildfire-sumatera-dataset
METADATA_LANDSAT_8_FILE_PATH  = f"{WILDFIRE_SUMATERA_DATASET_FOLDER_PATH}\\metadata_landsat_8.csv"
METADATA_SENTINEL_2_FILE_PATH = f"{WILDFIRE_SUMATERA_DATASET_FOLDER_PATH}\\metadata_sentinel_2.csv"



# Folders inside D:\\wildfire-sumatera-dataset\\wildfire-sumatera-geotiff
SENTINEL_2_GEOTIFF_FOLDER_PATH = f"{WILDFIRE_SUMATERA_GEOTIFF_FOLDER_PATH}\\sentinel-2"
LANDSAT_8_GEOTIFF_FOLDER_PATH  = f"{WILDFIRE_SUMATERA_GEOTIFF_FOLDER_PATH}\\landsat-8"


# Folders inside D:\\wildfire-sumatera-dataset\\wildfire-sumatera-jpeg
SENTINEL_2_JPEG_FOLDER_PATH = f"{WILDFIRE_SUMATERA_JPEG_FOLDER_PATH}\\sentinel-2"
LANDSAT_8_JPEG_FOLDER_PATH  = f"{WILDFIRE_SUMATERA_JPEG_FOLDER_PATH}\\landsat-8"



# Folders inside D:\\wildfire-sumatera-dataset\\wildfire-sumatera-geotiff\\landsat-8
LANDSAT_8_PREFIRE_GEOTIFF_FOLDER_PATH  = f"{LANDSAT_8_GEOTIFF_FOLDER_PATH}\\prefire"
LANDSAT_8_POSTFIRE_GEOTIFF_FOLDER_PATH = f"{LANDSAT_8_GEOTIFF_FOLDER_PATH}\\postfire"

# Folders inside D:\\wildfire-sumatera-dataset\\wildfire-sumatera-geotiff\\sentinel-2
SENTINEL_2_PREFIRE_GEOTIFF_FOLDER_PATH  = f"{SENTINEL_2_GEOTIFF_FOLDER_PATH}\\prefire"
SENTINEL_2_POSTFIRE_GEOTIFF_FOLDER_PATH = f"{SENTINEL_2_GEOTIFF_FOLDER_PATH}\\postfire"


# Folders inside D:\\wildfire-sumatera-dataset\\wildfire-sumatera-jpeg\\landsat-8
# LANDSAT_8_PREFIRE_JPEG_FOLDER_PATH  = f"{LANDSAT_8_JPEG_FOLDER_PATH}\\prefire"
# LANDSAT_8_POSTFIRE_JPEG_FOLDER_PATH = f"{LANDSAT_8_JPEG_FOLDER_PATH}\\postfire"

# Folders inside D:\\wildfire-sumatera-dataset\\wildfire-sumatera-jpeg\\sentinel-2
# SENTINEL_2_PREFIRE_JPEG_FOLDER_PATH  = f"{SENTINEL_2_JPEG_FOLDER_PATH}\\prefire"
# SENTINEL_2_POSTFIRE_JPEG_FOLDER_PATH = f"{SENTINEL_2_JPEG_FOLDER_PATH}\\postfire"


dirs = [
    WILDFIRE_SUMATERA_DATASET_FOLDER_PATH,
    WILDFIRE_SUMATERA_GEOTIFF_FOLDER_PATH, 
    WILDFIRE_SUMATERA_JPEG_FOLDER_PATH,
    
    SENTINEL_2_GEOTIFF_FOLDER_PATH, 
    LANDSAT_8_GEOTIFF_FOLDER_PATH,
    SENTINEL_2_JPEG_FOLDER_PATH,
    LANDSAT_8_JPEG_FOLDER_PATH,
    
    LANDSAT_8_PREFIRE_GEOTIFF_FOLDER_PATH,
    LANDSAT_8_POSTFIRE_GEOTIFF_FOLDER_PATH,
    SENTINEL_2_PREFIRE_GEOTIFF_FOLDER_PATH,
    SENTINEL_2_POSTFIRE_GEOTIFF_FOLDER_PATH,
    
#     LANDSAT_8_PREFIRE_JPEG_FOLDER_PATH,
#     LANDSAT_8_POSTFIRE_JPEG_FOLDER_PATH,
#     SENTINEL_2_PREFIRE_JPEG_FOLDER_PATH,
#     SENTINEL_2_POSTFIRE_JPEG_FOLDER_PATH,
]

for dir_ in dirs:
    if not os.path.exists(dir_):
        os.mkdir(dir_)
        print(f"{dir_} has been created")
    else:
        print(f"{dir_} already exist")