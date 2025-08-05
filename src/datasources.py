import os
import torch
from glob import glob

# ------------------------------------------------------ #
#                  Sentinel2   - S2
#                  Level 2A   - L2A
#                  Level 1C   - L1C
# ------------------------------------------------------ #

# Source: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/composites/
# NOTE: these are 1-indexed because GDAL is 1-indexed.
S2_ALL_12BANDS = {
    "true_color": [4, 3, 2],
    "false_color": [8, 4, 3],
    "swir": [12, 8, 4],
    "agriculture": [11, 8, 2],
    "geology": [12, 11, 2],
    "bathimetric": [4, 3, 1],
    "true_color_zero_index": [3, 2, 1],
}

S2_ALL_BANDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
METADATA_PATH = 'dataset/metadata.csv'

# Precomputed on training data by running: src/datasets.py
JIF_S2_MEAN = torch.Tensor(
    [
        0.0008,
        0.0117,
        0.0211,
        0.0198,
        0.0671,
        0.1274,
        0.1449,
        0.1841,
        0.1738,
        0.1803,
        0.0566,
        -0.0559,
    ]
).to(torch.float64)

JIF_S2_STD = torch.Tensor(
    [
        0.0892,
        0.0976,
        0.1001,
        0.1169,
        0.1154,
        0.1066,
        0.1097,
        0.1151,
        0.1122,
        0.1176,
        0.1298,
        0.1176,
    ]
).to(torch.float64)

# ------------------------------------------------------ #
#   Satellite pour l'Observation de la Terre - SPOT 7
#   Julien, Ivan, Freddie (working name)     - JIF
# ------------------------------------------------------ #

# HR
SPOT_BANDS = [0, 1, 2, 3]
SPOT_RGB_BANDS = [1, 2, 3]
SPOT_MAX_EXPECTED_VALUE_8_BIT = 255
SPOT_MAX_EXPECTED_VALUE_12_BIT = 10000

ROOT_JIF_DATA = os.path.join("data", "JIF")
ROOT_JIF_DATA_TRAIN = os.path.join(ROOT_JIF_DATA, "train")

URBAN_SPOT_MEAN = torch.Tensor([0.0025, 0.0022, 0.0017]).to(torch.float64)
URBAN_SPOT_STD = torch.Tensor([0.0021, 0.0018, 0.0017]).to(torch.float64)
