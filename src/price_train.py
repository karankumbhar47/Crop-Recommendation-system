import pandas as pd
from PricePipeline import PriceDataPipeline
from PriceModel import PriceModel

raw_prefix = "dataset/raw/crop-price/"
processed_prefix = "dataset/processed/crop-price/"
model_prefix = "models/"
crops = ['Maize', 'Rice', 'Wheat', 'Potatoes', 'Jowar', 'Soyabeans', 'Sweet_potatoes', 'Mango', 'Yams']
location_df = "dataset/location.csv"

for crop in crops:
    try:
        open(f"{processed_prefix}{crop}.csv")
        print(f"Data for {crop} is already processed...")
    except Exception:
        print(f"Processing data for {crop}...")
        df = pd.read_csv(f"{raw_prefix}{crop}.csv")
        pos = pd.read_csv(location_df)
        pipe = PriceDataPipeline(locations=pos)
        out = pipe.ProcessData(df, save_to=f"{processed_prefix}{crop}.csv")

# training models
for crop in crops:
    print(f"Training model for {crop}...")
    model = PriceModel(trained = False, file=f"{processed_prefix}{crop}.csv", show_logs=True)
    model.train(save_as=f"{model_prefix}{crop}.h5", epochs=20)
