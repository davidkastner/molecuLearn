"""Process ab-initio molecular dynamics simulations."""

import pandas as pd
from Bio.PDB import PDBParser, calc_mass, calc_center_of_mass
from itertools import combinations


def read_trajectory_pdb(file_path):
    """
    Read a trajectory PDB file and separate it into individual frames.
    
    Parameters
    ----------
    file_path : str
        Path to the PDB trajectory file.
    
    Returns
    -------
    list
        A list of PDB structures representing individual frames.

    """
    with open(file_path, 'r') as f:
        content = f.read()
    pdb_strings = content.split("END")
    parser = PDBParser()
    frames = [parser.get_structure("Frame", pdb_string) for pdb_string in pdb_strings if pdb_string.strip()]
    return frames
