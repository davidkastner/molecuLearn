"""Command-line interface (CLI) entry point."""

def welcome():
    """Print first to welcome the user while it waits to load the modules"""
    print("\n")
    print("            ╔═══════════════════════════════╗        ")
    print("            ║  ______ _____   ___           ║        ")
    print("            ║ |\   __ \ __  \|\  \          ║        ")
    print("            ║ \ \  \ \__\ \  \ \  \         ║        ")
    print("            ║  \ \  \|__|  \  \ \  \        ║        ")
    print("            ║   \ \  \    \ \  \ \  \____   ║        ")
    print("            ║    \ \__\    \ \__\ \_______\ ║        ")
    print("            ║     \|__|     \|__|\|_______| ║        ")
    print("            ║                               ║        ")
    print("            ║                               ║        ")
    print("            ║      MOLECULEARN (ML)         ║        ")
    print("            ║   [moleculearn.rtfd.io]       ║        ")
    print("            ╚═══════════════╗╔══════════════╝        ")
    print("                    ╔═══════╝╚═══════╗               ")
    print("                    ║ THE KULIK LAB  ║               ")
    print("                    ╚═══════╗╔═══════╝               ")
    print("  ╔═════════════════════════╝╚══════════════════════╗")
    print("  ║ Code: github.com/davidkastner/moleculearn       ║")
    print("  ║ Docs: moleculearn.readthedocs.io                ║")
    print("  ║    - RF model: ml -rf                           ║")
    print("  ║    - MLP model: ml -mlp                         ║")
    print("  ╚═════════════════════════════════════════════════╝\n")

# Greet the user with the welcome screen
welcome()

import os
import click

@click.command()
@click.option("--pairwise_distances", "-pd", is_flag=True, help="Compute pairwise distances")
@click.option("--combine_qm_charges", "-cq", is_flag=True, help="Combine charge data across QM data")
@click.option("--pairwise_features", "-pf", is_flag=True, help="Create pairwise features")
@click.option("--final_charge_dataset", "-fq", is_flag=True, help="Calculate final charge datasest")
@click.option("--calc_esp", "-ce", is_flag=True, help="Calculates ESP from Multiwfn output")
@click.option("--charge_esp_dataset", "-qe", is_flag=True, help="Creates dataset with ESP features")
@click.option("--random_forest", "-rf", is_flag=True, help="Run RF workflow")
@click.option("--mlp", "-mp", is_flag=True, help="Run MLP workflow")
@click.help_option("--help", "-h", is_flag=True, help="List of all moleculearn key words")
def cli(
    pairwise_distances,
    combine_qm_charges,
    final_charge_dataset,
    pairwise_features,
    calc_esp,
    charge_esp_dataset,
    random_forest,
    mlp,
    ):
    
    if pairwise_distances:
        click.echo("> Compute pairwise distances workflow:")
        click.echo("> Loading...")
        import ml.process
        
        click.echo("   > Combine the xyz files from all the single points:")
        replicate_info = ml.process.combine_sp_xyz()

        click.echo("   > Convert an xyz to a pdb trajectory:")
        ml.process.xyz2pdb_traj()

        click.echo("   > Compute pairwise distances features for a trajectory:")
        geometry_name = os.getcwd().split("/")[-1]
        infile = f"{geometry_name}_geometry.pdb"
        outfile = f"{geometry_name}_pairwise_distance.csv"
        mutations = [2,19,23]
        caps = [0,15,16,27]
        remove = sorted(mutations + caps)
        ml.process.check_file_exists(f"{geometry_name}_geometry.pdb")
        ml.process.pairwise_distances_csv(infile, outfile, replicate_info, remove)

    elif combine_qm_charges:
        click.echo("> Combining the QM charge data across single points:")
        click.echo("> Loading...")
        import ml.process

        compute_replicates = input("> Would you like this performed across replicates (y/n)? ")
        if compute_replicates == "n":
            ml.process.combine_qm_replicates()
        elif compute_replicates == "y":
            ml.process.combine_qm_charges(0, 39901, 100)
        else:
            print(f"> {compute_replicates} is not a valid response.")

    elif final_charge_dataset:
        click.echo("> Create final charge data set:")
        click.echo("> Loading...")
        import ml.process

        # Remove non-shared amino acids and caps
        mutations = [2,19,23]
        click.echo(f"> Non-shared residues are: {mutations}")
        caps = [0,15,16,27]
        click.echo(f"> Terminal caps are: {caps}")
        remove = sorted(mutations + caps)
        charges_df = ml.process.final_charge_dataset("all_charges.xls", "template.pdb", remove)

    elif pairwise_features:
        click.echo("> Generate pairwise charge data:")
        click.echo("> Loading...")
        import ml.process
        
        mimochrome_name = os.getcwd().split("/")[-1]
        # mimochrome_name = input("> Which mimochrome to process? ")
        charge_data = f"{mimochrome_name}_charges_ml.csv"
        ml.process.pairwise_charge_features(mimochrome_name, charge_data)

    elif calc_esp:
        click.echo("> Computed charge schemes with Multiwfn:")
        click.echo("> Loading...")
        import ml.process
        first = 0
        last = 39901
        step = 100
        ml.process.collect_esp_components(first, last, step)

    elif charge_esp_dataset:
        click.echo("> Create a charge dataset that contains ESP-derived features:")
        click.echo("> Loading...")
        import ml.process

        # Remove non-shared amino acids and terminal caps (zero index)
        mutations = [2,19,22]
        caps = [0,15,16,27]
        remove = sorted(mutations + caps)

        geometry_name = os.getcwd().split("/")[-1]
        charges_df = ml.process.final_charge_dataset("all_charges.xls", "template.pdb", remove)
        esp_scheme = input("> What ESP scheme would you like to add? ").capitalize()
        geometry_name = os.getcwd().split("/")[-1]
        ml.process.add_esp_charges(charges_df, esp_scheme, geometry_name)


    elif random_forest:
        click.echo("> Run Random Forest model on the cleaned data:")
        click.echo("> Loading...")
        import ml.rf
        
        # 1 splits each traj train/val/test; 2 splits all train/val/test
        data_split_type = int(input("   > Intra- (1) or inter-trajectory (2) data split? "))
        include_esp = input("   > Include ESP features (T/F)? ")

        # Pre-optimized hyperparameter sets named as model_split_ESP
        # [0] for distance and [1] for charge
        if data_split_type == 1 and include_esp == "T":       
            hyperparams = [{"max_depth":None, "mins_samples_leaf":3, "min_samples_split":3, "n_estimators":55},
                           {"max_depth":35, "mins_samples_leaf":3, "min_samples_split":6, "n_estimators":100}]   
        elif data_split_type == 1 and include_esp == "F":
            hyperparams = [{"max_depth":None, "mins_samples_leaf":3, "min_samples_split":3, "n_estimators":55},
                           {"max_depth":None, "mins_samples_leaf":3, "min_samples_split":3, "n_estimators":190}]    
        elif data_split_type == 2 and include_esp == "T":                
            hyperparams = [{"max_depth":None, "mins_samples_leaf":3, "min_samples_split":3, "n_estimators":55},
                           {"max_depth":40, "mins_samples_leaf":3, "min_samples_split":4, "n_estimators":105}] 
        elif data_split_type == 2 and include_esp == "F":                
            hyperparams = [{"max_depth":None, "mins_samples_leaf":2, "min_samples_split":2, "n_estimators":45},
                           {"max_depth":20, "mins_samples_leaf":2, "min_samples_split":2, "n_estimators":215}] 
        else:
            click.echo("No precomputed hyperparameters for this option.")

        ml.rf.rf_analysis(data_split_type, include_esp, hyperparams)
        # ml.rf.hyperparam_opt(data_split_type) # Uncomment and comment previous for hyperopt


    elif mlp:
        click.echo("> Run MLP model on the cleaned data:")
        click.echo("> Loading...")
        import ml.mlp
        
        # 1 splits each traj train/val/test; 2 splits all train/val/test
        data_split_type = int(input("   > Intra- (1) or inter-trajectory (2) data split? "))
        epochs = int(input("   > Epochs to run? "))
        include_esp = input("   > Include ESP features (T/F)? ")

        # Pre-optimized hyperparameter sets named as model_split_ESP
        # [0] for distance and [1] for charge
        if data_split_type == 1 and include_esp == "T":
            hyperparams = [{"lr":4.24E-05, "l2":1.01E-06, "n_layers":2, "n_neurons":40},
                           {"lr":1.73E-03, "l2":0.001853586, "n_layers":2, "n_neurons":148}]
        elif data_split_type == 1 and include_esp == "F":            
            hyperparams = [{"lr":0.001178084, "l2":2.51E-06, "n_layers":2, "n_neurons":214},
                           {"lr":0.000808712, "l2":0.009929909, "n_layers":2, "n_neurons":157}]
        elif data_split_type == 2 and include_esp == "T":
            hyperparams = [{"lr":2.56E-06, "l2":7.92E-04, "n_layers":2, "n_neurons":77},
                           {"lr":2.40E-05, "l2":0.001334242, "n_layers":2, "n_neurons":127}]
        elif data_split_type == 2 and include_esp == "F":
            hyperparams = [{"lr":2.445e-06, "l2":3.949e-06, "n_layers":2, "n_neurons":88},
                           {"lr":2.6575e-06, "l2":0.00567, "n_layers":3, "n_neurons":136}]
        else:
            click.echo("No precomputed hyperparameters for this option.")

        ml.mlp.run_mlp(data_split_type, include_esp, epochs, hyperparams)


    else:
        click.echo("No functionality was requested.\nTry --help.")


if __name__ == "__main__":
    # Run the command-line interface when this script is executed
    cli()
