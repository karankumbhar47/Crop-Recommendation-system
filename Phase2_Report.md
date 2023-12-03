# <center>Crop Recommendation System Report</center>

**[Project Link](https://github.com/karankumbhar47/Crop-Recommendation-system)**
(Link is at the end of Report)
## 1. Introduction

The Crop Recommendation System aims to assist farmers in making informed decisions about crop selection based on predicted crop yield and market prices. This system leverages two primary models: the Crop Yield Prediction model and the Price Predictor model. The Crop Yield Prediction model estimates the potential yield of various crops based on environmental and agricultural factors. Simultaneously, the Price Predictor model forecasts the market prices of these crops. By combining the yield and price predictions, the system recommends crops that are expected to yield high profits for the farmers.

## 2. Overall Idea

### 2.1 Crop Yield Prediction Model
The Crop Yield Prediction model utilizes Regression methods to forecast the potential yield of crops. It takes into account various features such as year, average rainfall, pesticide usage, average temperature, area, and crop type. The model is trained on historical data to learn patterns and relationships between these factors and crop yield. The output of this model is an estimation of the expected yield for a given set of environmental and agricultural conditions.

### 2.2 Price Predictor Model
The Price Predictor model forecasts the market prices of crops based on historical pricing data and relevant features. It considers factors such as arrival date, latitude, and longitude to predict the minimum, maximum, and modal prices for each crop. The model aims to provide better price predictions, `enabling farmers to anticipate market conditions`.

### 2.3 Crop Recommendation
The Crop Recommendation System combines the predictions from both models to recommend crops to farmers. The recommendation is based on maximizing the multiplication value of predicted yield and predicted prices. In other words, the system suggests crops that are anticipated to have both high yields and favorable market prices. This approach aims to optimize farmers' profits by considering both production and market conditions.

## 3. Crop Recommendation Process

The overall process of the Crop Recommendation System involves the following steps:

1. **Crop Yield Prediction:**
   - Input environmental and agricultural features.
   - Utilize the Crop Yield Prediction model to estimate potential crop yield.

2. **Price Prediction:**
   - Forecast market prices for the predicted yield using the Price Predictor model.

3. **Crop Recommendation:**
   - Calculate the multiplication value of predicted yield and price for each crop.
   - Recommend crops with the highest multiplication value, indicating potential profitability.

4. **Output:**
   - Recommends crop with most favorable conditions. And also provide farmers with a list of crops, along with estimated yield and market prices.

`Our Two Models are as follows :-` 

# Crop Yield Prediction

## 1. Introduction

The Crop Yield Prediction model aims to forecast the yield of crops based on diverse features, including year, average rainfall, pesticide usage, average temperature, area, and crop type. This report provides an overview of the machine learning models trained for the task, a comparison of their performance, and insights obtained from the evaluation.

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
   - Mean Absolute Error (MAE) and R-squared (R2) score are used to assess model accuracy. We got better MAE and R2 score for Decision Tree Regressor than other models.

c. **Variance:**
   - Decision Tree models with different depths and sample splits are compared for variance.By varying this two hyper-parameters we got that at max-depth 15 we got more accuracy. And second parameter will not change more change in accuracy.

d. **Training Time:**
   - Evaluate the training time for each model. But here, this four models training time is very less compared to other machine learning technique.

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

In conclusion, the Crop Yield Prediction model was trained and evaluated using various machine learning algorithms. The Decision Tree Regressor with specific hyperparameters demonstrated better results. Considerations regarding model complexity, accuracy, and training time were discussed. Further improvements could involve exploring ensemble methods, feature engineering, reliable dataset availability and additional hyperparameter tuning.

The trained model and preprocessor have been saved as 'Crop_Yield_Prediction.pkl' and 'preprocessor.pkl,' respectively.

---

# Price Predictor


## 1. Models Trained:
   - Recurrent Neural Network (RNN)


## 2. Model Comparison:

### a. Complexity, Ease of Interpretation:
   - The RNN model has a moderate level of complexity due to the presence of
     LSTM layers. While RNNs are generally more complex, the sequential nature
     of time series data makes them suitable. However, RNNs can be challenging
     to interpret compared to simpler models.

### b. Accuracy or Relevant Performance Metrics:
   - The model's performance was evaluated using mean squared error as the loss
     function. The choice of error metric is suitable for regression tasks like
     price prediction. The model achieved satisfactory accuracy on the
     validation set, as indicated by the mean absolute error.

### c. Variance:
   - Variance in this context refers to the variability of predictions across
     different instances. The model may exhibit variance due to factors such as
     market fluctuations and external influences. Regularization techniques and
     additional features could be explored to address variance.

### d. Training Time:
   - Training time for the RNN model is influenced by factors like the size of
     the dataset, the complexity of the model, and the number of training
     epochs. `The model was trained for a reasonable number of epochs to
     capture temporal patterns.`


## 3. Comparison with Research Papers:
   - While research papers might provide benchmarks, the effectiveness of
     models depends on the specific dataset and preprocessing steps. The
     implemented RNN model aligns with the general approach in research papers
     for time series forecasting. However, performance improvements can be
     sought by considering the following areas:
     - **Feature Engineering:** Explore additional relevant features that might impact crop prices, such as weather data, economic indicators, or seasonal trends.
     - **Hyperparameter Tuning:** Experiment with different hyperparameter configurations for the RNN, including the number of LSTM units, learning rate, and batch size.
     - **Ensemble Methods:** Combine predictions from multiple models or different architectures to enhance overall predictive power.
     - **Data Augmentation:** Generate additional training samples through data augmentation techniques, especially when dealing with a limited dataset.

## Conclusion:
The implemented RNN model serves as a reasonable baseline for crop price prediction. Further refinement and experimentation with features and model configurations could lead to improved accuracy and generalization. Additionally, incorporating domain-specific knowledge and insights can enhance the model's predictive capabilities.

---

## Scope for Improvement.

We can also imporve the result for corp recommendation by adding extra information which is cost for crop, But due to lack or reliable dataset , we won't able to implement it. So further we can improve it by adding this information also.


**Authors :-**
- Shubham Daule [12141540]
- Karan Kumbhar [12140860] 

**Project Link :-**  https://github.com/karankumbhar47/Crop-Recommendation-system
