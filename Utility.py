"""This module provides routines for setting the computing platform, precision etc

"""

import jax
import os


def SetPrecision(precision: str):
    
    jax.config.update("jax_enable_x64", True)


def SetPlatform(platform: str):
    
    if platform.upper() == "CPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        jax.config.update("jax_platform_name", "cpu")
