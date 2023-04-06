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

def pairwise_distances_csv(pdb_trajectory_path):
    """
    Generates a csv containing all the pairwise distances for a PDB trajectory.

    Parameters
    ----------
    pdb_trajectory_path: str
        The path to the PDB trajectory file

    """
    
    # Read and separate the PDB trajectory into frames
    frames = read_trajectory_pdb(pdb_trajectory_file)
    
    # Calculate pairwise distances for each frame and store them in a DataFrame
    pairwise_distances_df = trajectory_pairwise_distances(frames)
    
    # Save the DataFrame to a CSV file
    pairwise_distances_df.to_csv("pairwise_distances.csv", index=False)


if __name__ == "__main__":
    # Execute when run as a script
    pairwise_distances_csv("path/to/your/pdb_trajectory_file.pdb")
