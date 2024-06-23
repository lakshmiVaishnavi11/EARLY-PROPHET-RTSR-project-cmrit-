from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the machine learning model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        # Get the input values from the form
        duration = request.form['duration']
        departure_time = request.form['departure_time']
        arrival_time = request.form['arrival_time']
        date_of_journey = request.form['date_of_journey']
        total_stops = request.form['total_stops']
        airline = request.form['airline']
        source = request.form['source']
        destination = request.form['destination']
        route = request.form['route']

        # Preprocess the input values
        duration = convert_duration(duration) # type: ignore
        departure_time = pd.to_datetime(departure_time) # type: ignore
        arrival_time = pd.to_datetime(arrival_time) # type: ignore
        date_of_journey = pd.to_datetime(date_of_journey) # type: ignore
        total_stops = map_total_stops(total_stops) # type: ignore
        airline = pd.get_dummies(airline, drop_first=True) # type: ignore
        source = pd.get_dummies(source, drop_first=True) # type: ignore
        destination = pd.get_dummies(destination, drop_first=True) # type: ignore
        route = preprocess_route(route) # type: ignore

        # Create a numpy array from the input values
        input_values = np.array([duration, departure_time, arrival_time, date_of_journey, total_stops, airline, source, destination, route])

        # Make a prediction using the machine learning model
        prediction = model.predict(input_values)

        # Return the prediction result
        return render_template('index.html', predict_text='The predicted price is: ${:.2f}'.format(prediction[0]))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
