import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
import pickle as pkl

# Load the dataset
def read_original_data():
    yield_dataset_path = "./dataset/raw/Crop-yield-dataset/crop_yeild_prediction.csv"
    yield_data = pd.read_csv(yield_dataset_path)
    yield_data["Item"] = yield_data["Item"].replace("Sorghum", "Jowar")
    yield_data["Item"] = yield_data["Item"].replace("Rice, paddy", "Rice")
    yield_data["Item"] = yield_data["Item"].replace("Plantains and others", "Mango")
    yield_data["Item"] = yield_data["Item"].replace("Soybeans","Soyabeans") 

    return yield_data

# Separate the data into train and test sets
def separate_data(data, random_number=12140860):
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=random_number)
    train_data.reset_index(inplace=True, drop=True)
    test_data.reset_index(inplace=True, drop=True)
    return train_data, test_data

# Preprocess the data
def preprocess_data(train_data, test_data):
    global preprocesser
    col = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
    train_data = train_data[col]
    test_data = test_data[col]

    encoder = OneHotEncoder(drop='first')
    scaler = StandardScaler()

    preprocesser = ColumnTransformer(
        transformers=[
            ('StandardScale', scaler, [0, 1, 2, 3]),
            ('OHE', encoder, [4, 5]),
        ],
        remainder='passthrough'
    )

    X_train_dummy = preprocesser.fit_transform(train_data.iloc[:, :-1])
    X_test_dummy = preprocesser.transform(test_data.iloc[:, :-1])

    return X_train_dummy, X_test_dummy, train_data['hg/ha_yield'], test_data['hg/ha_yield']

# Train the final model
def train_final_model(X_train_dummy, X_test_dummy, y_train):
    final_model = DecisionTreeRegressor(min_samples_split=5, max_depth=10)
    final_model.fit(X_train_dummy, y_train)
    y_prediction = final_model.predict(X_test_dummy)
    print(f"Final Model: Mae : {mean_absolute_error(y_test, y_prediction)} Score : {r2_score(y_test, y_prediction)}")

    with open('Crop_Yield_Prediction.pkl', 'wb') as model_file:
        pkl.dump(final_model, model_file)

    pkl.dump(preprocesser, open('preprocessor.pkl', 'wb'))

if __name__ == "__main__":
    # Read the original data
    yield_data = read_original_data()

    # Separate data into train and test sets
    train_data, test_data = separate_data(yield_data)

    # Preprocess the data
    X_train_dummy, X_test_dummy, y_train, y_test = preprocess_data(train_data, test_data)

    # Train the final model
    train_final_model(X_train_dummy, X_test_dummy, y_train)
