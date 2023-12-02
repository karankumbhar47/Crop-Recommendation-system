import pandas as pd
import numpy as np
from datetime import datetime
import argparse

class CostDataPipeline():
    def __init__(self,
                 locations_file: str = None,
                 locations: pd.DataFrame = None):

        if locations_file is not None and locations is not None:
            raise Exception("Specify either `locations_file` or `locations` not both")
        if locations_file is None and locations is None:
            raise Exception("Specify either `locations_file` or `locations`")
        if locations is not None:
            self.locations = locations
        else:
            self.locations = pd.read_csv(locations_file)
        self.inputColumns = ["state", "district", "market", "commodity", "variety", "arrival_date", "min_price", "max_price", "modal_price", "update_date"]

    def get_coords(self,
                   state,
                   district,
                   location_data=None,
                   print_invalid=False
                  ) -> tuple[float, float]:

        if location_data is None:
            location_data = self.locations

        i = location_data[location_data["District"]==district]
        i = i[i["State"]==state.upper()]

        if i.shape[0] != 1:
            if print_invalid:
                print(f"{state},{district},{i.shape[0]}\n")
            return (np.nan, np.nan)

        return (float(i["Latitude"].iloc[0]), float(i["Longitude"].iloc[0]))

    def convert_date(self, date):
        return datetime.strptime(date, "%d/%m/%Y").toordinal()

    def ProcessData(self,
                    input_df: pd.DataFrame,
                    save_to: str = None,
                    dropna: bool = True,
                   ) -> pd.DataFrame:

        # Verify that the labels match
        if (input_df.columns != self.inputColumns).any():
            raise Exception(f"Columns of dataframe do not align, following are expected {self.inputColumns}")

        # Make a copy of data
        outputDf = input_df.copy()

        # Process Data
        coords = [self.get_coords(e["state"], e["district"]) for _,e in outputDf.iterrows()]
        lats = [coord[0] for coord in coords]
        longs = [coord[1] for coord in coords]
        outputDf = outputDf.assign(latitude=lats).assign(longitude=longs).drop(["state","district","market","variety", "update_date"],axis=1)

        outputDf["arrival_date"] = outputDf["arrival_date"].apply(self.convert_date)
        if dropna:
            outputDf.dropna(inplace=True)

        if save_to is not None:
            outputDf.to_csv(save_to, sep=",", index=False)

        return outputDf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Input file name')
    parser.add_argument('location_data', help='Locations data file')
    parser.add_argument('--save', help='Output file name')

    args = parser.parse_args()

    df = pd.read_csv(args.file)
    pos = pd.read_csv(args.location_data)
    pipe = CostDataPipeline(locations=pos)
    out = pipe.ProcessData(df, save_to=args.save)
    print(out)

if __name__ == "__main__":
    main()

