# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 23:49:44 2022

@author: DELL
"""

import numpy as np
import pickle
import streamlit as st
 
# loading the saved model
loaded_model = pickle.load(open('C:/Users/DELL/Desktop/FYP Project/trained_model.sav', 'rb'))

#Creating a funtion for prediction

def cost_prediction(input_data):
    
    #'Length', 'Transactions', 'Entities','PointsNonAdjust','PointsAjust'

    #input_data = (12,253,52,305, 302)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)


    if (prediction> 0):
        return 'Cost of the Prediction: ', prediction
    else:
        return 'Something Went Wrong'
    
    
    
    
    
    
    
    
    
    
def main():
    
    
    # giving a title
    st.title('Software Cost Prediction')
    
    
    # getting the input data from the user
    
    
    Length = st.number_input('Enter Number of Length')
    Transactions = st.number_input('Enter Number of Transaction')
    Entities = st.number_input('Enter the Number of Entities')
    PointsNonAdjust = st.number_input('Enter the Number of PointsNonAdjust')
    PointsAdjust = st.number_input('Enter the Number of PointsAdjust')
    
    
    # code for Prediction
    cost = ''
    
    # creating a button for Prediction
    
    if st.button('Cost Prediction'):
        cost = cost_prediction([Length, Transactions, Entities,PointsNonAdjust,PointsAdjust])
        
        
    st.success(cost)
    
    
    
    
    
if __name__ == '__main__':
    main()