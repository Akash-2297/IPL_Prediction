import streamlit as st
import pickle
import numpy as np
import pandas as pd

teams =['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities=['Hyderabad', 'Bangalore', 'Mumbai', 'Kolkata', 'Delhi', 'Indore',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']
pipe = pickle.load(open('pipe.pkl','rb'))
st.title('IPL Win predictor')

col1,col2 =st.columns(2)

with col1:
    batting_team=st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team=st.selectbox('Select bowling team',sorted(teams))
select_city=st.selectbox('Select city',sorted((cities)))

target=st.number_input('Target')
col3,col4,col5 =st.columns(3)
with col3:

    score=st.number_input('Score')
with col4:
    over=st.number_input('Over completed')
with col5:
    wicket=st.number_input("wicket")


if st.button('Predict probability'):
    run_left=target - score
    ball_left=120-(over)*6
    wicket=10-wicket
    crr=score/over
    rrr=(run_left*6)/ball_left
    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'balls_left': [ball_left],'city': [select_city],
                             'runs_left': [run_left],  'wickets': [wicket],
                             'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})
    result=pipe.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1]

    st.header(batting_team + "- " + str(round(win * 100)) + "%")
    st.header(bowling_team + "- " + str(round(loss * 100)) + "%")
    st.write('Required run rate',rrr)
    st.write('current run rate', crr)


