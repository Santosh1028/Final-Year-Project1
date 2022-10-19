import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('C:/Users/DELL/Desktop/FYP Project/trained_model.sav', 'rb'))

#'Length', 'Transactions', 'Entities','PointsNonAdjust','PointsAjust'

input_data = (12,253,52,305, 302)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)


if (prediction> 0):
    print('Cost of the Prediction: ', prediction)
else:
    print('Error Occured')