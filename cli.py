"""Command-line interface (CLI) entry point."""

# Print first to welcome the user while it waits to load the modules
print("\n.-------------------------------------.")
print("| WELCOME TO THE MOLECULEARN (ML) CLI |")
print(".-------------------------------------.")
print("Default programmed actions for the molecuLearn package.")
print("GitHub: https://github.com/davidkastner/moleculearn")
print("Documenation: https://moleculearn.readthedocs.io\n")

import click

@click.command()
@click.option("--pairwise_distances", "-d", is_flag=True, help="Compute pairwise distances.")
@click.option("--random_forest", "-rf", is_flag=True, help="Run RF workflow.")
@click.option("--mlp", "-mlp", is_flag=True, help="Run MLP workflow.")
def cli(
    pairwise_distances,
    random_forest,
    mlp,
    ):
    """
    The overall command-line interface (CLI) entry point.
    The CLI interacts with the rest of the package.

    A complete reference of molecuLearn functionality.
    This is advantagous because it quickly introduces so molecuLearn.
    Specificaly, to the complete scope of available functionality.
    It also improves long-term maintainability and readability.

    """
    
    if pairwise_distances:
        click.echo("> Compute pairwise distances features for a trajectory:")
        click.echo("> Loading...")
        import ml.process
        import ml.manage
        
        # Compute the pairwise distances
        pdb_traj_path = input("   > What is the name of your PDB? ")
        print(f"   > Assuming the PDB trajectory has name {pdb_traj_path}")
        ml.manage.check_file_exists(pdb_traj_path)
        ml.process.pairwise_distances_csv(pdb_traj_path)

    if random_forest:
        click.echo("> Run Random Forest model on the cleaned data:")
        click.echo("> Loading...")
        import ml.rf

        # Get datasets
        ml.rf.format_plots()
        mimos = ['mc6', 'mc6s', 'mc6sa']
        data_loc = input("   > Where are your data files located? ")
        df_charge, df_dist = ml.rf.load_data(mimos, data_loc)
        ml.rf.plot_data(df_charge, df_dist, mimos)

        # Preprocess the data and split into train and test sets
        data_split = ml.rf.preprocess_data(df_charge, df_dist, mimos)

        # Train a random forest classifier for each feature
        rf_cls = ml.rf.train_random_forest(data_split, n_trees=200, max_depth=50)

        # Evaluate classifiers and plot confusion matrices
        cms = ml.rf.evaluate(rf_cls, data_split, mimos)
        ml.rf.plot_confusion_matrices(cms, mimos)

    if mlp:
        click.echo("> Run MLP model on the cleaned data:")
        click.echo("> Loading...")
        import ml.mlp

        # Get datasets
        ml.mlp.format_plots()
        mimos = ["mc6", "mc6s", "mc6sa"]
        data_loc = input("   > Where are your data files located? ")
        df_charge, df_dist = ml.mlp.load_data(mimos, data_loc)
        ml.mlp.plot_data(df_charge, df_dist, mimos)

        # Preprocess the data and split into train, validation, and test sets
        data_split = ml.mlp.preprocess_data(df_charge, df_dist, mimos, 1)

        # Build the train, validation, and test dataloaders
        train_loader, val_loader, test_loader = ml.mlp.build_dataloaders(data_split)

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

        mlp_cls, train_loss_per_epoch, val_loss_per_epoch =ml.mlp.train(layers, 1e-3, 100, train_loader, val_loader, 'cpu')
        ml.mlp.plot_train_val_losses(train_loss_per_epoch, val_loss_per_epoch)
        test_loss, y_true, y_pred_proba, y_pred, cms = ml.mlp.evaluate_model(mlp_cls, test_loader, 'cpu', mimos)
        ml.mlp.plot_roc_curve(y_true, y_pred_proba, mimos)
        ml.mlp.plot_confusion_matrices(cms, mimos)

    else:
        click.echo("No functionality was requested.\nTry --help.")


if __name__ == "__main__":
    # Run the command-line interface when this script is executed
    cli()
