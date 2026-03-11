import colorsys
from typing import List, Tuple


def add_unique_key(d: dict, base_key: str, value) -> str:
    """
    Insert 'value' into dict 'd' using a unique key based on 'base_key'.
    The first insertion uses base_key_0, then base_key_1, etc.
    Returns the unique key used for insertion.
    
    :param d: The dictionary to insert into.
    :type d: dict
    :param base_key: The base string for the key.
    :type base_key: str
    :param value: The value to insert.
    
    :return: The unique key used for insertion.
    :rtype: str
    """
    i = 0
    new_key = f"{base_key}_{i}"
    while new_key in d:
        i += 1
        new_key = f"{base_key}_{i}"

    d[new_key] = value
    return new_key


def distinct_colors(n: int, s: float = 0.7, v: float = 0.9) -> List[Tuple[float, float, float]]:
    """
    Generate a list of visually distinct RGB colors.
    Colors are created by uniformly sampling the Hue component in HSV space
    and converting the result to RGB, This ensures maximal perceptual 
    separation between colors.
    
    :param n: The number of distinct colors to generate.
    :type n: int
    :param s: The saturation of the colors.
    :type s: float
    :param v: The value (brightness) of the colors.
    :type v: float
    
    :return: A list of RGB color tuples.
    :rtype: List[Tuple[float, float, float]]
    """
    return [
        colorsys.hsv_to_rgb(i / n, s, v)
        for i in range(n)
    ]
