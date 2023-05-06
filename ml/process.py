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

def combine_qm_replicates() -> None:
    """
    Combine the all_charges.xls files for replicates into a master charge file.

    The combined file contains a single header with atom numbers as columns.
    Each row represents a new charge instance.
    The first column indicates which replicate the charge came from.

    """
    start_time = time.time()  # Used to report the executation speed
    charge_file = "all_charges.xls"
    ignore = ["Analysis/"]

    # Directory containing all replicates
    primary_dir = os.getcwd()
    directories = sorted(glob.glob("*/"))
    replicates = [i for i in directories if i not in ignore]
    replicate_count = len(replicates)  # Report to user

    # Remove any old version because we are appending
    if os.path.exists(charge_file):
        os.remove(charge_file)
        print(f"      > Deleting old {charge_file}.")

    # Create a new file to save charges
    with open(charge_file, "a") as new_charge_file:
        header_written = False
        for replicate in replicates:
            # There will always be an Analysis folder
            os.chdir(replicate)
            secondary_dir = os.getcwd()
            print(f"   > Adding {secondary_dir}")

            # Get the replicate number from the folder name
            replicate_number = os.path.basename(os.path.normpath(secondary_dir))

            # Add the header for the first replicate
            with open(charge_file, "r") as current_charge_file:
                for index, line in enumerate(current_charge_file):
                    if index == 0:
                        if not header_written:
                            new_charge_file.writelines(line.strip() + "\treplicate\n")
                            header_written = True
                        continue
                    elif "nan" in line:
                        print(f"      > Found nan values in {secondary_dir}.")
                    else:
                        new_charge_file.writelines(line.strip() + "\t" + replicate_number + "\n")

            os.chdir(primary_dir)

    total_time = round(time.time() - start_time, 3)  # Seconds to run the function
    print(
        f"""
        \t----------------------------ALL RUNS END----------------------------
        \tRESULT: Combined charges across {replicate_count} replicates.
        \tOUTPUT: Generated {charge_file} in the current directory.
        \tTIME: Total execution time: {total_time} seconds.
        \t--------------------------------------------------------------------\n
        """
    )

def combine_replicates(
    all_charges: str = "all_charges.xls", all_coors: str = "all_coors.xyz"
) -> None:
    """
    Collects charges or coordinates into a xls and xyz file across replicates.

    Parameters
    ----------
    all_charges : str
        The name of the file containing all charges in xls format.
    all_coors.xyz : str
        The name of the file containing the coordinates in xyz format.

    Notes
    -----
    Run from the directory that contains the replicates.
    Run combine_restarts first for if each replicated was run across multiple runs.
    Generalized to combine any number of replicates.

    See Also
    --------
    qa.process.combine_restarts: Combines restarts and should be run first.
    """

    # General variables
    start_time = time.time()  # Used to report the executation speed
    files = [all_charges, all_coors]  # Files to be concatonated
    charge_files: list[str] = []  # List of the charge file locations
    coors_files: list[str] = []  # List of the coors file locations
    root = os.getcwd()
    dirs = sorted(glob.glob(f"{root}/*/"))  # glob to efficiently grab only dirs
    replicates = len(dirs)  # Only used to report to user

    # Loop through all directories containing replicates
    for dir in dirs:
        if os.path.isfile(f"{dir}{files[0]}") and os.path.isfile(f"{dir}{files[1]}"):
            charge_files.append(f"{dir}{files[0]}")
            coors_files.append(f"{dir}{files[1]}")

    new_file_names = [f"raw_{all_charges}", all_coors]
    file_locations = [charge_files, coors_files]
    # Loop over the file names and their locations
    for file_name, file_location in zip(new_file_names, file_locations):
        # Open a new file where we will write the concatonated output
        with open(file_name, "wb") as outfile:
            for loc in file_location:
                with open(loc, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)

    # The combined charge file now has multiple header lines
    first_line = True
    with open(new_file_names[0], "r") as raw_charge_file:
        with open(files[0], "w") as clean_charge_file:
            for line in raw_charge_file:
                # We want the first line to have the header
                if first_line == True:
                    clean_charge_file.write(line)
                    first_line = False
                # After the first, no lines should contain atom names
                else:
                    if "H" in line:
                        continue
                    else:
                        clean_charge_file.write(line)
    # Delete the charge file with the extra headers to keep the dir clean
    os.remove(new_file_names[0])

    total_time = round(time.time() - start_time, 3)  # Seconds to run the function
    print(
        f"""
        \t----------------------------ALL RUNS END----------------------------
        \tRESULT: Combined {replicates} replicates.
        \tOUTPUT: Generated {files[0]} and {files[1]} in the current directory.
        \tTIME: Total execution time: {total_time} seconds.
        \t--------------------------------------------------------------------\n
        """
    )

def combine_qm_charges(first_job: int, last_job: int, step: int) -> None:
    """
    Combines the charge_mull.xls files generate by TeraChem single points.

    After running periodic single points on the ab-initio MD data,
    we need to process the charge data so that it matches the SQM data.
    This code gets the charges from each single point and combines them.
    Results are stored in a tabular form.

    Parameters
    ----------
    first_job: int
        The name of the first directory and first job e.g., 0
    last_job: int
        The name of the last directory and last job e.g., 39901
    step: int
        The step size between each single point.

    """
    start_time = time.time()  # Used to report the executation speed
    new_charge_file = "all_charges.xls"
    current_charge_file = "charge_mull.xls"
    ignore = ["Analysis/"]

    # Directory containing all replicates
    primary_dir = os.getcwd()
    directories = sorted(glob.glob("*/"))
    replicates = [i for i in directories if i not in ignore]
    replicate_count = len(replicates)  # Report to user

    for replicate in replicates:
        frames = 0  # Saved to report to the user
        os.chdir(replicate)
        # The location of the current qm job that we are appending
        secondary_dir = os.getcwd()
        print(f"   > Adding { secondary_dir}")

        # Create a new file where we will store the combined charges
        first_charges_file = True  # We need the title line but only once

        if os.path.exists(new_charge_file):
            os.remove(new_charge_file)  # Since appending remove old version
            print(f"      > Deleting old {secondary_dir}/{new_charge_file}.")
        with open(new_charge_file, "a") as combined_charges_file:
            # A list of all job directories assuming they are named as integers
            job_dirs = [str(dir) for dir in range(first_job, last_job, step)]

            # Change into one of the QM job directories
            for index, dir in enumerate(job_dirs):
                os.chdir(dir)
                tertiary_dir = os.getcwd()
                os.chdir("scr")
                # Open an individual charge file from a QM single point
                atom_column = []
                charge_column = []

                # Open one of the QM charge single point files
                with open(current_charge_file, "r") as charges_file:
                    # Separate the atom and charge information
                    for line in charges_file:
                        clean_line = line.strip().split("\t")
                        charge_column.append(clean_line[1])
                        atom_column.append(clean_line[0])

                # Join the data and separate it with tabs
                charge_line = "\t".join(charge_column)

                # For some reason, TeraChem indexes at 0 with SQM,
                # and 1 with QM so we change the index to start at 1
                atoms_line_reindex = []
                for atom in atom_column:
                    atom_list = atom.split()
                    atom_list[0] = str(int(atom_list[0]) - 1)
                    x = " ".join(atom_list)
                    atoms_line_reindex.append(x)
                atom_line = "\t".join(atoms_line_reindex)

                # Append the data to the combined charges data file
                # We only add the header line once
                if first_charges_file:
                    combined_charges_file.write(f"{atom_line}\n")
                    combined_charges_file.write(f"{charge_line}\n")
                    frames += 1
                    first_charges_file = False
                # Skip the header if it has already been added
                else:
                    if "nan" in charge_line:
                        print(f"      > Found nan values in {index * 100}!!")
                    combined_charges_file.write(f"{charge_line}\n")
                    frames += 1

                os.chdir(secondary_dir)
        print(f"      > Combined {frames} frames.")
        os.chdir(primary_dir)

    total_time = round(time.time() - start_time, 3)  # Seconds to run the function
    print(
        f"""
        \t----------------------------ALL RUNS END----------------------------
        \tRESULT: Combined charges across {replicate_count} replicates.
        \tOUTPUT: Generated {new_charge_file} in the current directory.
        \tTIME: Total execution time: {total_time} seconds.
        \t--------------------------------------------------------------------\n
        """
    )

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

    return df


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

def calculate_esp(component_atoms, scheme):
    """
    Calculate the electrostatic potential (ESP) of a molecular component.

    Takes the output from a Multiwfn charge calculation and calculates the ESP.
    Run it from the folder that contains all replicates.
    It will generate a single csv file with all the charges for your residue,
    with one component/column specified in the input residue dictionary.

    Parameters
    ----------
    component_atoms: List[int]
        A list of the atoms in a given component

    """
    # Physical constants
    k = 8.987551 * (10**9)  # Coulombic constant in kg*m**3/(s**4*A**2)
    A_to_m = 10 ** (-10)
    KJ_J = 10**-3
    faraday = 23.06  # Kcal/(mol*V)
    C_e = 1.6023 * (10**-19)
    one_mol = 6.02 * (10**23)
    cal_J = 4.184
    component_esp_list = []

    # Open a charge scheme file as a pandas dataframe
    file_path = glob.glob(f"*_{scheme}.txt")[0]
    df_all = pd.read_csv(file_path, sep="\s+", names=["Atom", "x", "y", "z", "charge"])

    # The index of the metal center assuming iron Fe
    metal_index = df_all.index[df_all["Atom"] == "Fe"][0]
    component_atoms.append(metal_index)

    # Select rows corresponding to an atoms in the component
    df = df_all[df_all.index.isin(component_atoms)]
    df.reset_index(drop=True, inplace=True)

    # Get the new index of the metal as it will have changed
    metal_index = df.index[df["Atom"] == "Fe"][0]

    # Convert columns lists for indexing
    atoms = df["Atom"]  # Now contains only atoms in component
    charges = df["charge"]
    xs = df["x"]
    ys = df["y"]
    zs = df["z"]

    # Determine position and charge of the target atom
    xo = xs[metal_index]
    yo = ys[metal_index]
    zo = zs[metal_index]
    chargeo = charges[metal_index]
    total_esp = 0

    for idx in range(0, len(atoms)):
        if idx == metal_index:
            continue
        else:
            # Calculate esp and convert to units (A to m)
            r = (
                ((xs[idx] - xo) * A_to_m) ** 2
                + ((ys[idx] - yo) * A_to_m) ** 2
                + ((zs[idx] - zo) * A_to_m) ** 2
            ) ** 0.5
            total_esp = total_esp + (charges[idx] / r)

    # Note that cal/kcal * kJ/J gives 1
    component_esp = k * total_esp * ((C_e)) * cal_J * faraday

    return component_esp

def collect_esp_components(first_job: int, last_job: int, step: int) -> None:
    """
    Loops over replicates and single points and collects metal-centered ESPs.

    The main purpose is to navigagt the file structure and collect the data.
    The computing of the ESP is done in the calculate_esp() function.

    Parameters
    ----------
    first_job: int
        The name of the first directory and first job e.g., 0
    last_job: int
        The name of the last directory and last job e.g., 39900
    step: int
        The step size between each single point.

    See Also
    --------
    qa.process.calculate_esp()

    """
    start_time = time.time()  # Used to report the executation speed
    ignore = ["Analysis/", "coordinates/", "inputfiles/"]
    charge_schemes = ["ADCH", "Hirshfeld", "Mulliken", "Voronoi"]
    components = {
        "all": "1-487",
        "lower": "1-252",
        "upper": "253-424",
        "lower-his": "1-86,104-252",
        "heme": "425-486",
        "his": "87-103",
    }
    qm_job_count = 0

    # Directory containing all replicates
    primary_dir = os.getcwd()
    directories = sorted(glob.glob("*/"))
    replicates = [i for i in directories if i not in ignore]
    replicate_count = len(replicates)  # Report to user

    # Loop over each charge scheme which will be in the scr/ directory
    for scheme in charge_schemes:
        # Create a pandas dataframe with the columns from components keys
        charge_scheme_df = pd.DataFrame(columns=components.keys())

        # Loop over each replicate
        row_index = 0
        for replicate in replicates:
            os.chdir(replicate)

            # The location of the current qm job that we are appending
            secondary_dir = os.getcwd()

            # A list of all job directories assuming they are named as integers
            job_dirs = [str(dir) for dir in range(first_job, last_job, step)]

            # Change into one of the QM job directories
            for dir in job_dirs:
                os.chdir(dir)
                tertiary_dir = os.getcwd()
                os.chdir("scr/")
                row_index += 1

                # Loop of the values of our dictionary
                for key, value in components.items():
                    component_atoms = []
                    # Convert number strings, with commas and dashes, to numbers
                    for range_str in value.split(","):
                        start, end = map(int, range_str.split("-"))
                        component_atoms.extend(range(start - 1, end))

                        # Run a function
                        try:
                            component_esp = calculate_esp(
                                component_atoms, scheme
                            )
                        except:
                            print(f"Job: {replicate}-->{dir}")
                        charge_scheme_df.loc[row_index, key] = component_esp

                # Move back to the QM job directory
                qm_job_count += 1
                os.chdir(secondary_dir)

            os.chdir(primary_dir)

        # Save the dataframe to a csv file
        charge_scheme_df.to_csv(f"{scheme}_esp.csv", index=False)

    total_time = round(time.time() - start_time, 3)  # Time to run the function
    print(
        f"""
        \t----------------------------ALL RUNS END----------------------------
        \tRESULT: Performed operation on {replicate_count} replicates.
        \tOUTPUT: Output files for {qm_job_count} single points.
        \tTIME: Total execution time: {total_time} seconds.
        \t--------------------------------------------------------------------\n
        """
    )


def add_esp_charges(charges_df, esp_scheme, geometry_name):
    """
    Add the "upper" and "lower" columns from a CSV file to a DataFrame by removing
    and re-adding the "replicates" column.

    Parameters
    ----------
    charges_df : pandas.DataFrame
        The DataFrame to which the "upper" and "lower" columns will be added.
    esp_scheme : str
        The name of the esp file.
    geometry_name : str
        The name of the mimochrome or directory where we are working.

    Returns
    -------
    pandas.DataFrame
        The modified DataFrame with the "upper" and "lower" columns added
        and the "replicates" column re-added at the end.

    """
    # Read the csv file as a DataFrame
    csv_df = pd.read_csv(f"{esp_scheme}_esp.csv")

    # Select the "upper" and "lower" columns
    selected_columns = csv_df[["upper", "lower"]]

    # Remove the "replicates" column from charges_df and store it separately
    replicates_column = charges_df.pop("replicate")

    # Concatenate charges_df and selected_columns
    charges_df = pd.concat([charges_df, selected_columns], axis=1)

    # Add the "replicates" column back to charges_df
    charges_df["replicate"] = replicates_column
    charges_df.to_csv(f"{geometry_name}_charges_esp.csv", index=False)

    return charges_df


if __name__ == "__main__":
    # Execute when run as a script
    pairwise_distances_csv("path/to/your/pdb_trajectory_file.pdb")
