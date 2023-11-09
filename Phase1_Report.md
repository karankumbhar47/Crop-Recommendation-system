# Crop Recommendation System

- [Github Link]("https://github.com/karankumbhar47/Crop-Recommendation-system")

## Data collection and Preprocessing 

### Crop recommendation data
- This Dataset taken from [kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset), so it was already in state to use in our models. 
- For some analysis one hot encoding is used to convert strings values to boolean values.

### Crop price prediction data

- The data related to crop wise prices across different districts and states for a large number of days in a year was downloaded from [this](data.gov.in) website.
- The data uses names of states and districts when listing the prices. Although this is easy to understand, it does not carry any meaning for the machine. So we decided to replace this with the coordinates for the district. For this we downloaded a list of different districts and corrospondin coordinates. As the names of districts mentioned in the price data and coordinate data did not match, we manually cleaned this data and replaced the names with the latitudes.
- This dataset will be used to make RNN model which will be used to determine the expected price (and inturn profit) for the final model (see Final Task section).

## Basic model Training
- We have trained dataset over mulitple models state as follows
    1. Logistic Regression
    2. Random Forest
    3. Decision Tree
    4. KNN
    5. Gaussian Naive Bayes
    
- Among all the models that we have trained, we got highest training accuracy last model `Gaussian Naive Model` which is 99.05 precentage and cross validation accuracy nearly 99.45 percentage.
- Our project contains two models, in this phase we have completed first model `Basic Crop Recommendation Model`. Our Second model which we planned was about `Effect of crop on nutrient content of soil`. But we are unable to find appropriate data to train a machine learning model. Instead we have found direct results related to it. 

## Final Task Identification
- Our final task is to take input from user about soil condtion, environment factors related to location and  using this to recommend a crop. We will check the effect of repective crop on yeild, price, profit and soil condition.These will be calculated from harvesting location and date.

## Challenges
- **Lack of reliable dataset** - Our dataset used for crop recommendation model was not reliable. So we planned to make a dataset manually from government [website](data.gov.in). And train a model to predict a crop hueristically which will maximaize profit and soil quality.

- **Lack of dataset for soil quality model** - As mentioned above we are unable to find appropriate dataset for soil quality model, instead we got direct result for it. So we planned to use this data as look up table for second model.

## Final Deliverables
- We planned to make a webpage for taking input and display result from our model. 