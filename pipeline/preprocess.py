import click
import pandas as pd
from src.features.build_features import preprocess


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """Preprocess the raw data and save the processed data."""
    df = pd.read_csv(input_filepath)
    df_processed = preprocess(df)
    df_processed.to_csv(output_filepath, index=False)
    print(f"Processed data saved to {output_filepath}")


if __name__ == '__main__':
    main("", "")
