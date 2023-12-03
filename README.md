# Crop Recommendation System

This repository contains the Crop Recommendation System, a machine learning-based solution that recommends crops to farmers based on yield predictions and market prices.

## Table of Contents

- [Crop Recommendation System](#crop-recommendation-system)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training the Models](#training-the-models)
    - [Running Streamlit App](#running-streamlit-app)
    - [Test Model Prediction](#test-model-prediction)

## Introduction

The Crop Recommendation System combines two main models: Crop Yield Prediction and Price Prediction. It leverages machine learning to estimate crop yields based on environmental and agricultural factors and forecasts market prices for various crops. By maximizing the multiplication value of predicted yield and price, the system recommends crops expected to yield high profits.

## Installation

To use the Crop Recommendation System, follow these installation steps:

1. Clone the repository:

   ```bash
   git clone git@github.com:karankumbhar47/Crop-Recommendation-system.git 

   cd Crop-Recommendation-system
    ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Models

   To train the Crop Yield Prediction and Price Prediction models, use the provided scripts. Ensure you have the required datasets in the specified directories.

   ```bash
   python Yield_Model.py

   python price_train.py
   ```

### Running Streamlit App

  Launch the Streamlit web application to interact with the Crop Recommendation System.

  ```bash
  streamlit run src/web.py 
  ```

### Test Model Prediction 

1. Open the provided Streamlit web interface by navigating to http://localhost:8501 in your web browser.

2. Input the required information, such as environmental features and preferences.

3. Click the "Recommend" button to trigger the recommendation system.

4. Observe the recommended crops along with estimated income.

  Finally the recommeded crop with maximum estimated income will display on screen