import os

import click
import joblib

from src.models.train_model import train_model


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Runs the training process on the dataset provided at input_filepath
    and saves the trained model to output_filepath.
    """
    print(f"Training model using data from {input_filepath}")
    model = train_model(input_filepath)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # Save the model
    joblib.dump(model, output_filepath)
    print(f"Model saved to {output_filepath}")


if __name__ == "__main__":
    main()
