"""Command-line interface (CLI) entry point."""

# Print first to welcome the user while it waits to load the modules
print("\n.-------------------------------------------.")
print("| WELCOME TO THE MOLECULEARN (ML) CLI |")
print(".-------------------------------------------.")
print("Default programmed actions for the molecuLearn package.")
print("GitHub: https://github.com/davidkastner/moleculearn")
print("Documenation: https://moleculearn.readthedocs.io\n")

import click

@click.command()
@click.option("--pairwise_distances", "-d", is_flag=True, help="Compute pairwise distances.")
def cli(
    pairwise_distances,
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
        pdb_traj_path = "mc6_geometry.pdb"
        print(f"   > Assuming the PDB trajectory has name {pdb_traj_path}")
        ml.manage.check_file_exists(pdb_traj_path)
        ml.process.pairwise_distances_csv(pdb_traj_path)

    else:
        click.echo("No functionality was requested.\nTry --help.")


if __name__ == "__main__":
    # Run the command-line interface when this script is executed
    cli()
