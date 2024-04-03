import numpy as np
import pickle
def model8(dt):  
    try:
        to_predict_list = list(map(float, dt))
        lst = np.array(to_predict_list).reshape(1, 9)
        loaded_model = pickle.load(open("Models/water_potability.pkl", "rb"))
        result = loaded_model.predict(lst)
    except:
        result = -1
    return result