# <center>Crop Recommendation System Report</center>

**[Project Link](https://github.com/karankumbhar47/Crop-Recommendation-system)**
**Project Link :-**  https://github.com/karankumbhar47/Crop-Recommendation-system

## 1. Introduction

The Crop Recommendation System aims to assist farmers in making informed
decisions about crop selection based on predicted crop yield and expected
market prices. This system leverages two primary models: the Crop Yield
Prediction model and the Price Predictor model. The Crop Yield Prediction model
estimates the potential yield of various crops based on environmental and
agricultural factors. Whereas, the Price Predictor model predicts the
market prices of these crops. By combining the yield and price predictions, the
system recommends crops that are expected to yield highest profits for the
farmers.

## 2. Overall Idea

### 2.1 Crop Yield Prediction Model

The Crop Yield Prediction model utilizes Regression methods to forecast the
potential yield of crops. It takes into account various features such as year,
average rainfall, pesticide usage, average temperature, area, and crop type.
The model is trained on historical data to learn patterns and relationships
between these factors and crop yield. The output of this model is an estimation
of the expected yield for a given set of environmental and agricultural
conditions.

### 2.2 Price Predictor Model

The Price Predictor model forecasts the market prices of crops based on
location wise pricing data from the current past year. It considers factors
such as arrival date, latitude, and longitude to predict the minimum, maximum,
and modal prices for each crop. The output of this model is expected minimum,
maximum and modal price for the given crop.

### 2.3 Crop Recommendation

The Crop Recommendation System combines the predictions from both models to
recommend crops to the farmer. The recommendation is obtained by maximizing the
the expected income per month. This value of income is obtained by multiplying
the expected yeild and expected price. In other words, the system suggests
crops that are anticipated to have both high yields and favorable market
prices. This approach aims to optimize farmers' income by considering both
production and market conditions.

## 3. Crop Recommendation Process

The overall process of the Crop Recommendation System involves the following steps:

1. **Crop Yield Prediction:**
   - Input environmental and agricultural features.
   - Utilize the Crop Yield Prediction model to estimate potential crop yield.

2. **Price Prediction:**
   - Forecast market prices for the predicted yield using the Price Predictor model based on the location.

3. **Crop Recommendation:**
   - Calculate the expected income by multiplying yeild and price predicted for each crop.
   - Recommend crops with the highest estimated income, indicating potential profitability.

4. **Output:**
   - Recommends crop with most favorable weather and market conditions. And
     provide farmers with a list of crops, along with estimated income.

`Our Two Models are as follows :-` 

# Crop Yield Prediction

## 1. Introduction

The Crop Yield Prediction model aims to forecast the yield of crops based on
diverse features, including year, average rainfall, pesticide usage, average
temperature, area, and crop type. This report provides an overview of the
machine learning models trained for the task, a comparison of their
performance, and insights obtained from the evaluation.

## 2. Machine Learning Models

### 2.1 Models Trained

The following machine learning models were trained for the Crop Yield Prediction task:

- Linear Regression
- Lasso Regression
- Ridge Regression
- Decision Tree Regressor (with variations in hyperparameters)

## 3. Model Comparison

### 3.1 Comparison Criteria

The models are compared based on the following criteria:

a. **Complexity, Ease of Interpretation:**
   - Linear Regression is simple and highly interpretable.
   - Lasso and Ridge Regression introduce regularization, providing a balance between complexity and interpretability.
   - Decision Tree Regressor can be complex, especially with depth and sample split variations.

b. **Accuracy or Relevant Performance Metrics:**
   - Mean Absolute Error (MAE) and R-squared (R2) score are used to assess
     model accuracy. We got better MAE and R2 score for Decision Tree Regressor
     than other models.

c. **Variance:**
   - Decision Tree models with different depths and sample splits are compared
     for variance.By varying this two hyper-parameters we got that at max-depth
     15 we got more accuracy. And second parameter will not change more change
     in accuracy.

d. **Training Time:**
   - Evaluate the training time for each model. But here, this four models
     training time is very less compared to other machine learning technique.

### 3.2 Results

  #### Linear Regression
  - MAE: [~ 29907]
  - R2 Score: [0.74]

  #### Lasso Regression
  - MAE: [~ 29893]
  - R2 Score: [0.74]

  #### Ridge Regression
  - MAE: [~ 29863]
  - R2 Score: [0.74]

  #### Decision Tree Regressor
  - MAE: [~ 3935]
  - R2 Score: [0.97]

## 5. Conclusion

In conclusion, the Crop Yield Prediction model was trained and evaluated using
various machine learning algorithms. The Decision Tree Regressor with specific
hyperparameters demonstrated better results. Considerations regarding model
complexity, accuracy, and training time were discussed. Further improvements
could involve exploring ensemble methods, feature engineering, reliable dataset
availability and additional hyperparameter tuning.

The trained model and preprocessor have been saved as
'Crop_Yield_Prediction.pkl' and 'preprocessor.pkl,' respectively.

# Price Predictor

## 1. Models Trained:
   - Recurrent Neural Network (RNN)

## 2. Model Comparison:

### a. Complexity, Ease of Interpretation:

   - The RNN model has a moderate level of complexity due to the presence of
     LSTM layers. While RNNs are generally more complex, the sequential nature
     of price data makes them suitable.

### b. Accuracy or Relevant Performance Metrics:

   - The model's is trained on first 80% of the values and its performance is
     cross validated on the remaining 20% data using mean squared error as the
     loss function. The model achieved satisfactory accuracy on the validation
     set.

### c. Regularization and Hyperparameters:

   - L2 regularization was used on the kernel value to avoid overfitting.
   - The following hyperparameters showed based results, batch_size 16,
     sequence_length 5. These can be changed easily when initializing the
     PriceModel calss.
   - I used 20 epochs as there is no appearent reduction in loss after this
     point.

### d. Training Time:

   - The training time is quite small, about 5 to 7 seconds per epoch for every
     model.

## Conclusion:

The implemented RNN model serves as a reasonable baseline for crop price
prediction. Further refinement and experimentation with features and model
configurations could lead to improved accuracy and generalization.
Additionally, incorporating domain-specific knowledge and insights can enhance
the model's predictive capabilities.

---

## Scope for Improvement.

Our prediction does not currently take into account the factors like the cost
of production, maturity period, soil conditions, fertilizer usage etc. These
were not added due to lack of reliable dataset.

The price prediction model provides minimum, maximum and model price. We
currently only use the modal price prediction. The model can be modified to
incorporate the minimum and maximum prices as well in some form. We can provide
things like how much the price deviates from the modal price, etc.

The data from multiple years can be used in the price predictor model to make it
more accurate. The prices of a product during same time of the year might have
some relation which can be exploited to get better results.

**Authors :-**
- Shubham Daule [12141540]
- Karan Kumbhar [12140860] 

**Project Link :-**  https://github.com/karankumbhar47/Crop-Recommendation-system
