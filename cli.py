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
@click.option("--first_action", "-a", is_flag=True, help="Place holder description.")
@click.option("--second_action", "-b", is_flag=True, help="Place holder description.")
def cli(
    first_action,
    second_action,
    ):
    """
    The overall command-line interface (CLI) entry point.
    The CLI interacts with the rest of the package.

    A complete reference of quantumAllostery functionality.
    This is advantagous because it quickly introduces so quantumAllostery.
    Specificaly, to the complete scope of available functionality.
    It also improves long-term maintainability and readability.

    """
    
    if first_action:
        click.echo("> Description for user of what is being performed:")
        click.echo("> Loading...")
        import ml.process
        # Run a function from the molecuLearn package


    elif second_action:
        click.echo("> Combine trajectories from multiple replicates:")
        click.echo("> Loading...")
        import ml.plot
        # Run a function from the molecuLearn package


    else:
        click.echo("No functionality was requested.\nTry --help.")


if __name__ == "__main__":
    # Run the command-line interface when this script is executed
    cli()
