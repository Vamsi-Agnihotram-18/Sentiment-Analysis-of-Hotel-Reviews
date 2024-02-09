from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and vectorizer
with open(r'C:\Users\Sai Naveen Chanumolu\Desktop\DM\xgboost_model.pkl', 'rb') as file:
    xgboost_model = joblib.load(file)

vectorizer_filename = r'C:\Users\Sai Naveen Chanumolu\Desktop\DM\tfidf_vectorizer.pkl'
vectorizer = joblib.load(vectorizer_filename)

df = pd.read_csv(r'C:\Users\Sai Naveen Chanumolu\Desktop\DM\Hotel_Reviews.csv')

def preprocess_text(text):
    return text.lower()

def calculate_sentiment_score(reviews):
    processed_reviews = [preprocess_text(review) for review in reviews]
    review_vectors = vectorizer.transform(processed_reviews)
    predicted_sentiments = xgboost_model.predict(review_vectors)

    # Count the number of positive reviews (coded as '2')
    positive_reviews = sum(sentiment == 2 for sentiment in predicted_sentiments)

    # Calculate the proportion of positive reviews and scale it to 0-5
    if len(predicted_sentiments) > 0:
        sentiment_score = (positive_reviews / len(predicted_sentiments)) * 5
    else:
        sentiment_score = 0  # Handle case with no reviews

    return sentiment_score


@app.route('/')
def index():
    top_cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami']
    city_top_hotels = {}

    for city in top_cities:
        city_hotels = df[df['city'].str.lower() == city.lower()]

        # Filter hotels with at least 3 reviews
        hotels_with_enough_reviews = city_hotels.groupby('name').filter(lambda x: len(x) >= 3)

        # Calculate sentiment score for each eligible hotel
        hotel_sentiments = {}
        for hotel in hotels_with_enough_reviews['name'].unique():
            hotel_reviews = hotels_with_enough_reviews[hotels_with_enough_reviews['name'] == hotel]['reviews.text'].tolist()
            hotel_sentiments[hotel] = calculate_sentiment_score(hotel_reviews)

        # Sort hotels by sentiment score and select top 5
        top_hotels = sorted(hotel_sentiments.items(), key=lambda x: x[1], reverse=True)[:5]

        city_top_hotels[city] = [
            {
                'hotel': hotel[0],
                'sentiment_score': f"{hotel[1]:.2f}",
                'address': city_hotels[city_hotels['name'] == hotel[0]]['address'].iloc[0],
                'website': city_hotels[city_hotels['name'] == hotel[0]]['websites'].iloc[0]
            } for hotel in top_hotels
        ]

    return render_template('index.html', city_top_hotels=city_top_hotels)



@app.route('/hotel/<hotel_name>')
def hotel_detail(hotel_name):
    hotel_data = df[df['name'].str.lower() == hotel_name.lower()]
    if hotel_data.empty:
        return "Hotel not found", 404

    reviews = hotel_data['reviews.text'].tolist()
    avg_rating = hotel_data['reviews.rating'].mean()
    sentiment_score = calculate_sentiment_score(reviews)

    details = {
    'name': hotel_name,
    'avg_rating': f"{avg_rating:.1f}" if not hotel_data.empty else 'N/A',
    'sentiment_score': f"{sentiment_score:.2f}",
    'reviews': hotel_data[['reviews.text', 'reviews.rating']].values.tolist(),
    'website': hotel_data['websites'].iloc[0] if 'websites' in hotel_data.columns and not hotel_data['websites'].empty else '#',  # Comma added here
    'latitude': hotel_data['latitude'].iloc[0],
    'longitude': hotel_data['longitude'].iloc[0],
    'address': hotel_data['address'].iloc[0],

}


    return render_template('hotel_detail.html', details=details)
@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['input_text'].lower()

    # Initialize target_hotel_name to None
    target_hotel_name = None

    # Check if the input is a hotel name or city
    is_hotel = any(df['name'].str.lower() == user_input)
    is_city = any(df['city'].str.lower() == user_input)

    if is_hotel or is_city:
        # Determine the relevant set of hotels based on input
        if is_hotel:
            target_hotel_name = user_input  # Store the searched hotel name
            target_city = df[df['name'].str.lower() == target_hotel_name]['city'].iloc[0].lower()
        else:
            target_city = user_input

        similar_hotels = df[df['city'].str.lower() == target_city]

        # Filter similar hotels with at least 3 reviews
        similar_hotels_with_enough_reviews = similar_hotels.groupby('name').filter(lambda x: len(x) >= 3)

        # Calculate sentiment scores for similar hotels
        similar_hotels_grouped = similar_hotels_with_enough_reviews.groupby('name')
        sentiment_scores = {}
        for name, group in similar_hotels_grouped:
            sentiment_scores[name] = calculate_sentiment_score(group['reviews.text'].tolist())

        # Rank similar hotels based on their sentiment score
        ranked_similar_hotels = sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)

        # Recommend top N similar hotels
        top_n = 4
        recommended_hotels = [hotel[0] for hotel in ranked_similar_hotels[:top_n]]

        # Prepare the details to be displayed for each recommended hotel
        recommended_hotels_details = []
        for hotel in recommended_hotels:
            hotel_data = similar_hotels_with_enough_reviews[similar_hotels_with_enough_reviews['name'] == hotel]
            average_rating = hotel_data['reviews.rating'].mean()
            hotel_address = hotel_data['address'].iloc[0] if not hotel_data.empty else 'N/A'
            hotel_website = hotel_data['websites'].iloc[0] if not hotel_data.empty else '#'
            hotel_info = {
                'name': hotel,
                'sentiment_score': f"{sentiment_scores[hotel]:.2f}",
                'average_rating': f"{average_rating:.1f}" if not hotel_data.empty else 'N/A',
                'address': hotel_address,
                'website': hotel_website
            }
            recommended_hotels_details.append(hotel_info)
    else:
        return "No matching hotels or cities found in the dataset."

    # Add the searched hotel to the recommendations if it's not already in the list
    if target_hotel_name and target_hotel_name not in recommended_hotels:
        target_hotel_data = df[df['name'].str.lower() == target_hotel_name]
        target_average_rating = target_hotel_data['reviews.rating'].mean()
        target_sentiment_score = calculate_sentiment_score(target_hotel_data['reviews.text'].tolist())
        target_hotel_info = {
            'name': target_hotel_name,
            'sentiment_score': f"{target_sentiment_score:.2f}",
            'average_rating': f"{target_average_rating:.1f}",
            'address': target_hotel_data['address'].iloc[0] if not target_hotel_data.empty else 'N/A',
            'website': target_hotel_data['websites'].iloc[0] if not target_hotel_data.empty else '#'
        }
        recommended_hotels_details.insert(0, target_hotel_info)  # Insert at the beginning of the list

    return render_template('recommendations.html', recommendations=recommended_hotels_details)



if __name__ == '__main__':
    app.run(debug=True)
