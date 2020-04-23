import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology
from skimage import filters
from skimage import measure
from pyclustering.cluster import kmedoids
from typing import Tuple

from merlin.util import matlab

"""
This module contains utility functions for preparing imagmes for performing
segmentation using machine learning approaches
"""

def apply_machine_learning_segmentation(imageStackIn: np.ndarray,
										method: str) -> np.ndarray:

def segment_using_ilastik(imageStackIn: np.ndarray) -> np.ndarray:

def segment_using_unet(imageStackIn: np.ndarray) -> np.ndarray:

def segment_using_cellpose(imageStackIn: np.ndarray) -> np.ndarray:

