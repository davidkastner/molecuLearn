"""Command-line interface (CLI) entry point."""

# Print first to welcome the user while it waits to load the modules
print("\n.-------------------------------------.")
print("| WELCOME TO THE MOLECULEARN (ML) CLI |")
print(".-------------------------------------.")
print("Default programmed actions for the molecuLearn package.")
print("GitHub: https://github.com/davidkastner/moleculearn")
print("Documenation: https://moleculearn.readthedocs.io\n")

import os
import shutil
import click

@click.command()
@click.option("--pairwise_distances", "-pd", is_flag=True, help="Compute pairwise distances.")
@click.option("--combine_qm_charges", "-cq", is_flag=True, help="Combine charge data across QM single points.")
@click.option("--pairwise_charge_features", "-pq", is_flag=True, help="Create pairwise charge features.")
@click.option("--final_charge_dataset", "-fq", is_flag=True, help="Calculate final charge datasest.")
@click.option("--calc_esp", "-ce", is_flag=True, help="Calculates ESP from Multiwfn output.")
@click.option("--esp_dataset", "-esp", is_flag=True, help="Creates dataset with ESP features.")
@click.option("--random_forest", "-rf", is_flag=True, help="Run RF workflow.")
@click.option("--mlp", "-mlp", is_flag=True, help="Run MLP workflow.")
def cli(
    pairwise_distances,
    combine_qm_charges,
    final_charge_dataset,
    pairwise_charge_features,
    calc_esp,
    esp_dataset,
    random_forest,
    mlp,
    ):
    """
    The overall command-line interface (CLI) entry point.
    The CLI interacts with the rest of the package.

    A complete reference of molecuLearn functionality.
    This is advantagous because it quickly introduces molecuLearn.
    Specificaly, to the complete scope of available functionality.
    It also improves long-term maintainability and readability.

    """
    
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
        ml.process.check_file_exists(f"{geometry_name}_geometry.pdb")
        ml.process.pairwise_distances_csv(infile, outfile, replicate_info)

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
        mutations = [2,19,22]
        caps = [1,16,17,28]
        remove = sorted(mutations + caps)
        charges_df = ml.process.final_charge_dataset("all_charges.xls", "template.pdb", remove)

    elif pairwise_charge_features:
        click.echo("> Generate pairwise charge data:")
        click.echo("> Loading...")
        import ml.process
        
        mimochrome_name = os.getcwd().split("/")[-1]
        input_file = f"{mimochrome_name}_charges_ml.csv"
        ml.process.pairwise_charge_features(mimochrome_name, input_file)

    elif calc_esp:
        click.echo("> Computed charge schemes with Multiwfn:")
        click.echo("> Loading...")
        import ml.process
        first = 0
        last = 39901
        step = 100
        ml.process.collect_esp_components(first, last, step)

    elif esp_dataset:
        click.echo("> Create a charge dataset that contains ESP-derived features:")
        click.echo("> Loading...")
        import ml.process

        # Remove non-shared amino acids and terminal caps
        mutations = [2,19,22]
        caps = [1,16,17,28]
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

        # Get datasets
        ml.rf.format_plots()
        mimos = ['mc6', 'mc6s', 'mc6sa']
        data_loc = input("   > Where are your data files located (enter for cwd)? ") or os.getcwd()
        df_charge, df_dist = ml.rf.load_data(mimos, data_loc)
        ml.rf.plot_data(df_charge, df_dist, mimos)

        # Preprocess the data and split into train and test sets
        # data_split, df_dist, df_charge = ml.rf.preprocess_data(df_charge, df_dist, mimos, 2, test_frac=0.875)
        data_split, df_dist, df_charge = ml.rf.preprocess_data(df_charge, df_dist, mimos, 1)

        # Train a random forest classifier for each feature
        rf_cls = ml.rf.train_random_forest(data_split, n_trees=200, max_depth=50)

        # Evaluate classifiers and generate plots
        cms,y_true, y_pred_proba = ml.rf.evaluate(rf_cls, data_split, mimos)
        ml.rf.plot_roc_curve(y_true, y_pred_proba, mimos)
        ml.rf.plot_confusion_matrices(cms, mimos)
        ml.rf.shap_analysis(rf_cls, data_split, df_dist, df_charge, mimos)
        ml.rf.plot_gini_importance(rf_cls, df_dist, df_charge)

        # Clean up the newly generated files
        rf_dir = "RF"
        # Create the "rf/" directory if it doesn't exist
        if not os.path.exists(rf_dir):
            os.makedirs(rf_dir)

        # Move all files starting with "rf_" into the "rf/" directory
        for file in os.listdir():
            if file.startswith("rf_"):
                shutil.move(file, os.path.join(rf_dir, file))

    elif mlp:
        click.echo("> Run MLP model on the cleaned data:")
        click.echo("> Loading...")
        import ml.mlp

        # Get datasets
        ml.mlp.format_plots()
        mimos = ["mc6", "mc6s", "mc6sa"]
        data_loc = input("   > Where are your data files located (enter for cwd)? ") or os.getcwd()
        df_charge, df_dist = ml.mlp.load_data(mimos, data_loc)
        ml.mlp.plot_data(df_charge, df_dist, mimos)

        # Preprocess the data and split into train, validation, and test sets
        # data_split, df_dist, df_charge = ml.mlp.preprocess_data(df_charge, df_dist, mimos, 2, val_frac=0.75, test_frac=0.875)
        data_split, df_dist, df_charge = ml.mlp.preprocess_data(df_charge, df_dist, mimos, 1)

        # Build the train, validation, and test dataloaders
        train_loader, val_loader, test_loader = ml.mlp.build_dataloaders(data_split)

        # Get input sizes for each dataset and build model architectures
        n_dist = data_split['dist']['X_train'].shape[1]
        n_charge = data_split['charge']['X_train'].shape[1]
        layers = {'dist': (ml.mlp.torch.nn.Linear(n_dist, 128), ml.mlp.torch.nn.ReLU(), 
                           ml.mlp.torch.nn.Linear(128, 128), ml.mlp.torch.nn.ReLU(), 
                           ml.mlp.torch.nn.Linear(128, 128), ml.mlp.torch.nn.ReLU(), 
                           ml.mlp.torch.nn.Linear(128, 3)),
                'charge': (ml.mlp.torch.nn.Linear(n_charge, 128), ml.mlp.torch.nn.ReLU(), 
                           ml.mlp.torch.nn.Linear(128, 128), ml.mlp.torch.nn.ReLU(), 
                           ml.mlp.torch.nn.Linear(128, 128), ml.mlp.torch.nn.ReLU(), 
                           ml.mlp.torch.nn.Linear(128, 3))
                }
        
        # Train model on training and validation data
        mlp_cls, train_loss_per_epoch, val_loss_per_epoch =ml.mlp.train(layers, 1e-3, 100, train_loader, val_loader, 'cpu')
        ml.mlp.plot_train_val_losses(train_loss_per_epoch, val_loss_per_epoch)
        # Evaluate model on test data
        test_loss, y_true, y_pred_proba, y_pred, cms = ml.mlp.evaluate_model(mlp_cls, test_loader, 'cpu', mimos)
        # Plot ROC-AUC curves, confusion matrices and SHAP dot plots
        ml.mlp.plot_roc_curve(y_true, y_pred_proba, mimos)
        ml.mlp.plot_confusion_matrices(cms, mimos)
        ml.mlp.shap_analysis(mlp_cls, test_loader, df_dist, df_charge, mimos)

        # Clean up the newly generated files
        mlp_dir = "MLP"
        # Create the "rf/" directory if it doesn't exist
        if not os.path.exists(mlp_dir):
            os.makedirs(mlp_dir)

        # Move all files starting with "rf_" into the "rf/" directory
        for file in os.listdir():
            if file.startswith("mlp_"):
                shutil.move(file, os.path.join(mlp_dir, file))


    else:
        click.echo("No functionality was requested.\nTry --help.")


if __name__ == "__main__":
    # Run the command-line interface when this script is executed
    cli()
