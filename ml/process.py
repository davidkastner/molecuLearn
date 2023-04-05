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

def pairwise_distances(structure):
    """
    Calculate pairwise distances between the center of mass of amino acids in a PDB structure.
    
    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        The PDB structure object.
    
    Returns
    -------
    dict
        A dictionary with keys as tuples of residue pairs and values as their pairwise distances.

    """
    residues = list(structure.get_residues())
    residue_pairs = combinations(residues, 2)
    distances = {}
    for r1, r2 in residue_pairs:
        com1 = calc_center_of_mass(r1)
        com2 = calc_center_of_mass(r2)
        distance = calc_mass.distance(com1, com2)
        distances[(r1.get_resname(), r1.id[1], r2.get_resname(), r2.id[1])] = distance
    return distances

