"""Process ab-initio molecular dynamics simulations."""

import pandas as pd
import numpy as np
import io
from Bio.PDB import PDBParser, Vector, Atom
from itertools import combinations


def read_trajectory_pdb(file_path):
    """
    Read a PDB trajectory file and return a list of PDB structure objects.
    
    Parameters
    ----------
    file_path : str
        The path to the PDB trajectory file.
    
    Returns
    -------
    list
        A list of Bio.PDB.Structure.Structure objects, one for each frame in the trajectory.
    """
    with open(file_path, 'r') as f:
        pdb_string = f.read()
    pdb_strings = pdb_string.split("END")
    parser = PDBParser(QUIET=True)
    frames = [parser.get_structure("Frame", io.StringIO(pdb_string)) for pdb_string in pdb_strings if pdb_string.strip()]
    return frames

def center_of_mass(residue):
    """
    Calculate the center of mass of a residue.
    
    Parameters
    ----------
    residue : Bio.PDB.Residue.Residue
        A residue object.
    
    Returns
    -------
    tuple
        A tuple of (x, y, z) coordinates of the center of mass.
    """
    total_mass = 0
    mass_center = Vector(0, 0, 0)
    for atom in residue.get_atoms():
        atom_mass = Atom.atom_masses.get(atom.element, 12.0)  # default to 12.0 if element not in atom_masses dictionary
        mass_center += atom.vector * atom_mass
        total_mass += atom_mass
    return tuple(mass_center / total_mass)

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
        com1 = center_of_mass(r1)  # Use the custom center_of_mass function
        com2 = center_of_mass(r2)
        distance = np.linalg.norm(np.array(com1) - np.array(com2))
        distances[(r1.get_resname(), r1.id[1], r2.get_resname(), r2.id[1])] = distance
    return distances

def trajectory_pairwise_distances(frames):
    """
    Calculate pairwise distances for all frames in a PDB trajectory.
    
    Parameters
    ----------
    frames : list
        A list of PDB structures representing individual frames.
    
    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with rows as frames and columns as pairwise distances.

    """
    all_distances = [pairwise_distances(frame) for frame in frames]
    df = pd.DataFrame(all_distances)
    return df

def pairwise_distances_csv(pdb_traj_path):
    """
    Generates a csv containing all the pairwise distances for a PDB trajectory.

    Parameters
    ----------
    pdb_trajectory_path: str
        The path to the PDB trajectory file

    See Also
    --------
    read_trajectory_pdb()
    pairwise_distances()
    trajectory_pairwise_distances()

    """
    
    # Read and separate the PDB trajectory into frames
    frames = read_trajectory_pdb(pdb_traj_path)
    
    # Calculate pairwise distances for each frame and store them in a DataFrame
    pairwise_distances_df = trajectory_pairwise_distances(frames)
    
    # Save the DataFrame to a CSV file
    pairwise_distances_df.to_csv("pairwise_distances.csv", index=False)


if __name__ == "__main__":
    # Execute when run as a script
    pairwise_distances_csv("path/to/your/pdb_trajectory_file.pdb")
