import pandas as pd
import numpy as np
from datetime import datetime
import argparse

class CropYieldDataPipeline():
    def __init__(self,
                 locations_file: str = None,
                 locations: pd.DataFrame = None):

        if locations_file is not None and locations is not None:
            raise Exception("Specify either `locations_file` or `locations`, not both")
        if locations_file is None and locations is None:
            raise Exception("Specify either `locations_file` or `locations`")
        if locations is not None:
            self.locations = locations
        else:
            self.locations = pd.read_csv(locations_file)
        self.inputColumns = ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "Area", "Item", "hg/ha_yield"]

    def is_string(self, value):
        try:
            float(value)
            return False
        except:
            return True

    def convert_to_float(self, value):
        try:
            return float(value)
        except:
            return np.nan

    def process_data(self, input_df: pd.DataFrame, save_to: str = None) -> pd.DataFrame:
        output_df = input_df.copy()

        # Drop unwanted columns
        output_df.drop('Unnamed: 0', axis=1, inplace=True)

        # Remove duplicate values
        output_df.drop_duplicates(inplace=True)

        # Handle non-numeric strings in 'average_rain_fall_mm_per_year'
        output_df['average_rain_fall_mm_per_year'] = output_df['average_rain_fall_mm_per_year'].apply(self.convert_to_float)

        # Drop rows where 'average_rain_fall_mm_per_year' is not numeric
        dropped_rows = output_df[output_df['average_rain_fall_mm_per_year'].apply(self.is_string)].index
        output_df = output_df.drop(dropped_rows)

        # Additional preprocessing steps...

        output_df["Item"] = output_df["Item"].replace("Sorghum","Jowar") 
        output_df["Item"] = output_df["Item"].replace("Rice, paddy","Rice") 
        output_df["Item"] = output_df["Item"].replace("Plantains and others","Mango") 

        if save_to is not None:
            output_df.to_csv(save_to, sep=",", index=False)

        return output_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Input file name')
    parser.add_argument('locations', help='Locations data file')
    parser.add_argument('--save', help='Output file name')

    args = parser.parse_args()

    df = pd.read_csv(args.file)
    loc = pd.read_csv(args.locations)
    pipeline = CropYieldDataPipeline(locations=loc)
    processed_data = pipeline.process_data(df, save_to=args.save)
    print(processed_data)

if __name__ == "__main__":
    main()
