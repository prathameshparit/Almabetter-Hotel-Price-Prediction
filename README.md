# Hotel-Price-Prediction

![image](https://github.com/prathameshparit/Almabetter-Hotel-Price-Prediction/assets/63944541/93db2b75-e8d7-4286-b042-f6bf8e8c38e9)

# Travel Package Price Prediction

This project is designed to predict the price of travel packages based on various features and input data. It utilizes machine learning techniques to make accurate predictions.

## Setup

### Prerequisites

Before you can run this project, ensure you have the necessary libraries and models installed:

- Python 3
- Flask
- numpy
- sentence_transformers
- joblib
- pandas

You will also need a pre-trained SentenceTransformer model for text embeddings. You can initialize it using:

```python
model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')
```

### Model Loading

The project loads the following pre-trained models from the `./model` directory:

- Linear Regression model (`lr_model.joblib`)
- Principal Component Analysis (PCA) model (`pca.joblib`)
- Scaler for data transformation (`scaler.joblib`)

### Prediction

The core function for making predictions is `predict_price`, which takes input data and returns a price prediction.

## Usage

To use this project, you can run it locally. The main Flask app provides a web interface for inputting travel package details and getting price predictions.

```python
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

## Input Fields

The input form allows you to enter details about the travel package, such as package name, destination, itinerary, and more. You can also select various options like package type, travel month, and more.

## Web Interface

This project provides a simple web interface for interacting with the prediction model. It's accessible by visiting the root URL ("/") in your web browser.

## API Endpoint

You can also make predictions via API by sending a POST request to the "/predict" endpoint with the necessary data. The prediction is returned in JSON format.

## License

Feel free to customize and use this project for your own travel package price predictions.
