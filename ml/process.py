"""Process ab-initio molecular dynamics simulations."""

import glob
import os
import io
import time
import pandas as pd
import numpy as np
import MDAnalysis as mda
from typing import List
from biopandas.pdb import PandasPdb
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


def pairwise_distances_csv(pdb_traj_path, output_file, replicate_info):
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

    # Processes replicate info and add it as the last column
    replicate_list = []
    for replicate, count in replicate_info:
        replicate_list.extend([replicate] * count)
    replicate_col = pd.DataFrame(replicate_list, columns=['replicate'])

    # Ensure the replicate_col has the same number of rows as the other dataframe.
    assert len(pairwise_distances_df) == len(replicate_col), "Dataframes have different number of rows."

    # Concatenate the dataframes along the columns axis
    pairwise_distances_df = pd.concat([pairwise_distances_df, replicate_col], axis=1)
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


def pairwise_charge_features(structure):
    """
    Generate pairwise charge features for a given metalloenzyme structure.

    This function reads charge data from a CSV file, computes pairwise charge features
    using the specified operation (addition or multiplication), and saves the results
    as a new CSV file. The input CSV file should contain columns with charges for each
    amino acid in the metalloenzyme, with each row representing a frame from an
    ab-initio molecular dynamics simulation. The last column should be named "replicate"
    and contain information about the replicate each row belongs to.

    Parameters
    ----------
    structure : str
        The name of the metalloenzyme structure, used to find the input CSV file
        (named "{structure}_charges.csv") and to name the output CSV file
        (named "{structure}_charges_pairwise_{operation}.csv").

    Raises
    ------
    ValueError
        If the user inputs an invalid operation for pairwise charge features.
        Valid operations are "add" and "multiply".

    """
    # Load the data
    input_file = f"{structure}_charges.csv"
    df = pd.read_csv(input_file)

    # Remove the "replicate" column and store it separately
    replicate = df.pop("replicate")

    # Prompt the user for the desired operation
    operation = input("Choose an operation for pairwise charge features (add/multiply): ")

    # Generate pairwise charge features
    feature_columns = df.columns
    new_features = []
    for i, col1 in enumerate(feature_columns):
        for j, col2 in enumerate(feature_columns):
            if j <= i:  # Skip duplicate pairs
                continue

            # Perform the specified operation
            if operation == "add":
                new_features.append(pd.DataFrame({f"{col1}-{col2}": df[col1] + df[col2]}))
            elif operation == "multiply":
                new_features.append(pd.DataFrame({f"{col1}-{col2}": df[col1] * df[col2]}))
            else:
                raise ValueError("Invalid operation. Choose 'add' or 'multiply'.")

    # Concatenate the original DataFrame and new DataFrame
    df = pd.concat([df] + new_features, axis=1)

    # Add the "replicate" column back in as the final column
    df = pd.concat([df, replicate], axis=1)

    # Save the results as a new CSV
    output_file = f"{structure}_charges_pairwise_{operation}.csv"
    df.to_csv(output_file, index=False)

def get_residue_identifiers(template, by_atom=True) -> List[str]:
    """
    Gets the residue identifiers such as Ala1 or Cys24.

    Returns either the residue identifiers for every atom, if by_atom = True
    or for just the unique amino acids if by_atom = False.

    Parameters
    ----------
    template: str
        The name of the template pdb for the protein of interest.
    by_atom: bool
        A boolean value for whether to return the atom identifiers for all atoms

    Returns
    -------
    residues_indentifier: List(str)
        A list of the residue identifiers

    """
    # Get the residue number identifiers (e.g., 1)
    residue_number = (
        PandasPdb().read_pdb(template).df["ATOM"]["residue_number"].tolist()
    )
    # Get the residue number identifiers (e.g., ALA)
    residue_name = PandasPdb().read_pdb(template).df["ATOM"]["residue_name"].tolist()
    # Combine them together
    residues_indentifier = [
        f"{name}{number}" for number, name in zip(residue_number, residue_name)
    ]

    # Return only unique entries if the user sets by_atom = False
    if not by_atom:
        residues_indentifier = list(OrderedDict.fromkeys(residues_indentifier))

    return residues_indentifier


def summed_residue_charge(charge_data: pd.DataFrame, template: str):
    """
    Sums the charges for all atoms by residue.

    Reduces inaccuracies introduced by the limitations of Mulliken charges.

    Parameters
    ----------
    charge_data: pd.DataFrame
        A DataFrame containing the charge data.
    template: str
        The name of the template pdb for the protein of interest.

    Returns
    -------
    sum_by_residues: pd.DataFrame
        The charge data averaged by residue and stored as a pd.DataFrame.

    """
    # Extract the "replicate" column and remove it from the charge_data DataFrame
    replicate_column = charge_data['replicate']
    charge_data = charge_data.drop('replicate', axis=1)

    # Get the residue identifiers (e.g., 1Ala) for each atom
    residues_indentifier = get_residue_identifiers(template)

    # Assign the residue identifiers as the column names of the charge DataFrame
    charge_data.columns = residues_indentifier
    sum_by_residues = charge_data.groupby(by=charge_data.columns, sort=False, axis=1).sum()

    # Add the "replicate" column back to the sum_by_residues DataFrame
    sum_by_residues['replicate'] = replicate_column

    return sum_by_residues

def final_charge_dataset(charge_file: str, template: str, mutations: List[int]) -> pd.DataFrame:
    """
    Create final charge data set.

    The output from the combined charges is an .xls file with atoms as columns.
    We will combine the atoms by residues and average the charges.

    Returns
    -------
    charges_df: pd.DataFrame
        The original charge data as a pandas dataframe.

    """
    print(f"   > Converting atoms to residues for {charge_file}.")
    # Load the charge file as a DataFrame
    charge_data = pd.read_csv(charge_file, sep='\t')
    
    # Average atoms by residue to minimize the inaccuracies of Mulliken charges
    avg_by_residues = summed_residue_charge(charge_data, template)

    # Drop the residue columns that were mutated
    # We can't compare these residues' charges as their atom counts differ
    charges_df = avg_by_residues.drop(avg_by_residues.columns[[m for m in mutations]], axis=1)

     # Save the individual dataframe to a CSV file
    geometry_name = os.getcwd().split("/")[-1]
    output_file = f"{geometry_name}_charges.csv"
    charges_df.to_csv(output_file, index=False)
    print(f"   > Saved {output_file}.")

    return charges_df


if __name__ == "__main__":
    # Execute when run as a script
    pairwise_distances_csv("path/to/your/pdb_trajectory_file.pdb")
