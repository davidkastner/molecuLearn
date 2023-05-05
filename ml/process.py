"""Process ab-initio molecular dynamics simulations."""

import glob
import os
import io
import time
import pandas as pd
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from itertools import combinations

def combine_sp_xyz():
    """
    Combines single point xyz's for all replicates.

    The QM single points each of a geometry file.
    Combines all those xyz files into.
    Preferential to using the other geometry files to insure they are identical.

    Returns
    -------
    replicate_info : List[tuple()]
        List of tuples with replicate number and frame count for the replicates.

    """
    start_time = time.time()  # Used to report the executation speed

    # Get the directories of each replicate
    primary = os.getcwd()
    replicates = sorted(glob.glob("*/"))
    ignore = ["Analyze/", "Analysis/", "coordinates/", "inputfiles/", "opt-wfn/"]

    xyz_count = 0
    replicate_info = []  # Initialize an empty list to store replicate information

    # Get the name of the structure
    geometry_name = os.getcwd().split("/")[-1]
    out_file = f"{geometry_name}_geometry.xyz"

    with open(out_file, "w") as combined_sp:
        for replicate in replicates:
            if replicate in ignore:
                continue
            else:
                print(f"   > Adding replicate {replicate} structures.")
                os.chdir(replicate)
                secondary = os.getcwd()
                os.chdir("coordinates")

                structures = sorted(glob.glob("*.xyz"))
                frame_count = 0  # Initialize frame count for each replicate
                for index, structure in enumerate(structures):
                    with open(structure, "r") as file:
                        # Write the header from the first file
                        combined_sp.writelines(file.readlines())
                        xyz_count += 1
                        frame_count += 1

                replicate_info.append((int(replicate[:-1]), frame_count))  # Append replicate information

            # Go back and loop through all the other replicates
            os.chdir(primary)

    total_time = round(time.time() - start_time, 3)  # Time to run the function
    print(
        f"""
        \t----------------------------ALL RUNS END----------------------------
        \tOUTPUT: Combined {xyz_count} single point xyz files.
        \tOUTPUT: Output file is {out_file}.
        \tTIME: Total execution time: {total_time} seconds.
        \t--------------------------------------------------------------------\n
        """
    )

    return replicate_info 


def xyz2pdb_traj() -> None:
    """
    Converts an xyz trajectory file into a pdb trajectory file.

    Note
    ----
    Make sure to manually check the PDB that is read in.
    Assumes no header lines.
    Assumes that the only TER flag is at the end.

    """

    start_time = time.time()  # Used to report the executation speed

    # Get the name of the structure
    pdb_name = "template.pdb"
    geometry_name = os.getcwd().split("/")[-1]
    xyz_name = f"{geometry_name}_geometry.xyz"
    new_pdb_name = f"{geometry_name}_geometry.pdb"

    # Open files for reading
    xyz_file = open(xyz_name, "r").readlines()
    pdb_file = open(pdb_name, "r").readlines()
    max_atom = int(pdb_file[len(pdb_file) - 3].split()[1])
    new_file = open(new_pdb_name, "w")

    atom = -1  # Start at -1 to skip the XYZ header
    line_count = 0
    for line in xyz_file:
        line_count += 1
        if atom > 0:
            atom += 1
            try:
                x, y, z = line.strip("\n").split()[1:5]  # Coordinates from xyz file
                element_name = line.strip("\n").split()[0]
            except:
                print(f"> Script died at {line_count} -> '{line}'")
                quit()
            pdb_line = pdb_file[atom - 2].strip()  # PDB is two behind the xyz
            if len(element_name) > 1:
                new_file.write(f"{pdb_line[0:30]}{x[0:6]}  {y[0:6]}  {z[0:6]}  {pdb_line[54:80]}          {element_name}\n")
            else:
                new_file.write(f"{pdb_line[0:30]}{x[0:6]}  {y[0:6]}  {z[0:6]}  {pdb_line[54:80]}           {element_name}\n")
        else:
            atom += 1
        if atom > max_atom:
            atom = -1
            new_file.write("END\n")

    total_time = round(time.time() - start_time, 3)  # Seconds to run the function
    print(
        f"""
        \t----------------------------ALL RUNS END----------------------------
        \tRESULT: Converted {xyz_name} to {new_pdb_name}.
        \tOUTPUT: Generated {new_pdb_name} in the current directory.
        \tTIME: Total execution time: {total_time} seconds.
        \t--------------------------------------------------------------------\n
        """
    )


def pairwise_distances_csv(pdb_traj_path, output_file):
    """
    Calculate pairwise distances between residue centers of mass and save the result to a CSV file.
    
    Parameters
    ----------
    pdb_traj_path : str
        The file path of the PDB trajectory file.
    output_file : str
        The name of the output CSV file.
    """

    start_time = time.time()  # Used to report the executation speed

    # Read the trajectory file and split it into models
    with open(pdb_traj_path) as f:
        models = f.read().split("END")
    
    # Create a list of StringIO objects for each model
    frame_files = [io.StringIO(model) for model in models if model.strip()]
    universes = [mda.Universe(frame_file, format="pdb") for frame_file in frame_files]

    # Generate column names based on residue pairs
    residue_names = [residue.resname + str(residue.resid) for residue in universes[0].residues]
    residue_pairs = list(combinations(residue_names, 2))
    column_names = [f"{pair[0]}-{pair[1]}" for pair in residue_pairs]

    pairwise_distances = []
    for universe in universes:
        # Calculate the center of mass for each residue
        residue_com = np.array([residue.atoms.center_of_mass() for residue in universe.residues])
        
        # Calculate the pairwise distance matrix
        distance_matrix = distances.distance_array(residue_com, residue_com)
        pairwise_distances.append(distance_matrix[np.triu_indices(len(residue_com), k=1)])

    # Create a DataFrame with pairwise distances and column names
    pairwise_distances_df = pd.DataFrame(pairwise_distances, columns=column_names)
    pairwise_distances_df.to_csv(output_file, index=False)

    total_time = round(time.time() - start_time, 3)  # Seconds to run the function
    print(
        f"""
        \t----------------------------ALL RUNS END----------------------------
        \tRESULT: Created pairwise distance data set from xyz's.
        \tOUTPUT: Save pairwise distance data to {output_file}.
        \tTIME: Total execution time: {total_time} seconds.
        \t--------------------------------------------------------------------\n
        """
    )


if __name__ == "__main__":
    # Execute when run as a script
    pairwise_distances_csv("path/to/your/pdb_trajectory_file.pdb")
