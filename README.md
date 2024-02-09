#### Overview
This project is designed to analyze hotel reviews and predict their sentiment using machine learning models. It includes a web application for easy interaction with the models. The application allows users to input review text and receive sentiment predictions.

#### Contents
- `app.py`: The Flask application script that serves the web interface for interacting with the sentiment analysis models.
- `Hotel_Reviews.csv`: The dataset containing hotel reviews used to train the machine learning models.
- `logreg_model.pkl`: The Logistic Regression model trained to predict the sentiment of hotel reviews.
- `sentimentAnalysisofHotelReviews.ipynb`: A Jupyter notebook that documents the process of exploring the dataset, training the models, and evaluating their performance.
- `templates/`: A directory containing HTML templates for the Flask application, including the main interface for submitting review text.
- `tfidf_vectorizer.pkl`: The TF-IDF vectorizer used to convert review text into a format suitable for model input.
- `vectorizer.pkl`: An additional text vectorization model used for feature extraction from review texts.
- `xgboost_model.pkl`: An XGBoost model that offers an alternative to the Logistic Regression model for sentiment prediction.

#### Setup
To run this project, ensure you have Python installed along with the necessary libraries, including Flask, scikit-learn, and XGBoost. You can install dependencies using the following command:

```bash
pip install flask scikit-learn xgboost
```

#### Running the Application
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Run the Flask application with the following command:

```bash
python app.py
```

4. Open a web browser and navigate to `http://127.0.0.1:5000/` to interact with the application.

#### How It Works
- **Input**: The user submits a hotel review text through the web interface.
- **Processing**: The application uses the TF-IDF vectorizer to transform the input text into a vector. Then, it predicts the sentiment of the review using the trained Logistic Regression or XGBoost model.
- **Output**: The application displays the predicted sentiment (positive or negative) to the user.

#### Contributing
We welcome contributions to this project, including enhancements to the prediction models, improvements to the web interface, and additional documentation.
