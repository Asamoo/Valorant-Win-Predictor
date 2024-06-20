import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('casual_xgboost_model.pkl')

def plus_minus(plus, minus):
    return plus - minus

def predict(input_data):
    prediction = model.predict(np.array([input_data]))
    probability = model.predict_proba(np.array([input_data]))
    return prediction[0], probability[0]

labels = [
    'ACS', 'Kills', 'Deaths', 'Assists',
    'ADR', 'FK', 'Econ Rating', 'Plants', 'Defuses', 'Rounds',
    'Triple Kills', 'Quadra Kills', 'Penta Kills'
]

st.title('Valorant Win Probability Calculator')

st.write('This is a web application designed to predict whether or not somebody won a game of valorant based on their individual performance. The program allows you to input your ACS, kills, deaths, assists, ADR, FKs, Econ Rating, Plants, Defuses, Rounds played, and triple, quadra, and penta kills.')

st.write('For the uninitiated, here is a glossary of the terms from above:')

glossary_data = {
    'Term': [
        'ACS', 'Kills', 'Deaths', 'Assists', 'ADR', 'FK', 'Econ Rating', 'Plants', 'Defuses', 'Rounds',
        'Triple Kills', 'Quadra Kills', 'Penta Kills'
    ],
    'Definition': [
        "Valorant's in-game Average Combat Score",
        'Total number of kills',
        'Total number of deaths',
        'Total number of assists',
        'Average Damage per Round',
        'First Kills',
        "Valorant's in-game Econ Rating",
        'Number of times you planted the spike',
        'Number of times you defused the spike',
        'Total number of rounds played',
        'Number of rounds with exactly 3 kills',
        'Number of rounds with exactly 4 kills',
        'Number of rounds with exactly 5 kills'
    ]
}

glossary_df = pd.DataFrame(glossary_data)

st.table(glossary_df)

st.write("You can find your ACS, Kills, Deaths, Assists, FKs, Econ Ratings, Plants, and Defuses on the scoreboard of any competitive game you've played in your career. Valorant lists first kills as First Bloods. You'll find your ADR under the summary tab listed as Damage per Round.")
st.write("The summary tab also lists the number of triple, quadra, and penta kills you had. If you didn't get any triple, quadra, and/or penta kills, please input 0 there. If you had a round with more than 5 kills because of a Sage or Clove ultimate, you can treat them as a penta kill and add it to your penta kill total.")
st.write('Lastly, for rounds, please input the TOTAL number of rounds you played, not just the number of rounds you won. This means adding the number of rounds you lost to the number of rounds you won and entering that total value.')

st.write('The calculator below allows you to input a value for the above stats. It will predict whether or not you won and the probability that you won. The program functions in such a way that any probabilities below 50% are classified as a loss, while probabilities above 50% are classified as a win.')

input_values = []

cols_1 = st.columns(7)
cols_2 = st.columns(6)

for i in range(7):
    with cols_1[i]:
        input_value = st.text_input(f'{labels[i]}')
        input_values.append(input_value)

for i in range(7, 13):
    with cols_2[i - 7]:
        input_value = st.text_input(f'{labels[i]}')
        input_values.append(input_value)

if st.button('Classify'):
    if len(input_values) == 13:
        try:
            float_values = [float(value) for value in input_values]
            
            if any(value < 0 for value in float_values):
                raise ValueError('All input values must be non-negative.')

            int_fields = [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12]
            for idx in int_fields:
                if not float_values[idx].is_integer():
                    raise ValueError(f'{labels[idx]} must be an integer.')

            kills = float_values[1]
            deaths = float_values[2]
            kd_plus_minus = plus_minus(kills, deaths)
            
            triple_kills = float_values[10]
            quadra_kills = float_values[11]
            penta_kills = float_values[12]
            multikills = triple_kills + quadra_kills + penta_kills
            
            input_data = [
                float_values[0], kills, deaths, float_values[3],
                kd_plus_minus, float_values[4], float_values[5], multikills, float_values[6],
                float_values[7], float_values[8], float_values[9]
            ]
            
            prediction, probability = predict(input_data)

            result = 'Win' if prediction == 1 else 'Loss'
            
            st.write(f'Prediction: {result}')
            st.write(f'Probability: {round(probability[1], 4)}')
        except ValueError as e:
            st.write(f'Error: {e}')
    else:
        st.write('Please enter all 13 values.')
