"""Contains directory and file book keeping functions."""

import os
import sys
import numpy as np

def check_file_exists(filename):
    """
    Check if a file with the given name exists. If it does not exist, the function ends the current
    Python session with an informative error message.

    Parameters
    ----------
    filename : str
        The name of the file to check for existence.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.

    Examples
    --------
    >>> check_file_exists("example.txt")
    None

    >>> check_file_exists("non_existent_file.txt")
    FileNotFoundError: File 'non_existent_file.txt' does not exist. Please provide a valid file name.
    """

    # Check if the file exists
    if not os.path.isfile(filename):
        # Raise a FileNotFoundError with a custom error message
        raise FileNotFoundError(f"File '{filename}' does not exist. Please provide a valid file name.")
    else:
        print(f"File '{filename}' exists.")
        return None
