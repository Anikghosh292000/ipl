import pickle
import streamlit as st
st.title('IPL RUN PREDICTOR')

sixth_th_run = st.number_input('6th_run', value=70, placeholder='Enter run after 6th over')
first_st_wicket = st.number_input('1st_wicket', value=5, placeholder='Enter wicket after 6th over')
fortheen_th_over = st.number_input('14th_over', value=150, placeholder='Enter run after 14th over')
second_nd_wicket = st.number_input('2nd_wicket', value=9, placeholder='Enter wicket after 14th over')

loaded_model = pickle.load(open('ipl_pred.sav', 'rb'))
prediction =loaded_model.predict([[sixth_th_run, first_st_wicket, fortheen_th_over, second_nd_wicket]])

st.subheader(f'Predicted score for the above parameters is {int(prediction[0])}')
st.write(int(prediction[0]))
