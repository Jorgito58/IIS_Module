from joblib import load
import numpy as np
import pandas as pd

class IIS_Manager():

    def __init__(self) -> None:
        """
        Initializes the class. This is called by __init__ and should not be called directly. You can use it as a constructor
        """

        # Class
        self.Crop_Label = ['apple','banana','blackgram','chickpea','coconut','coffee',
                      'cotton','grapes','jute','kidneybeans','lentil','maize','mango',
                      'mothbeans','mungbean','muskmelon','orange','papaya','pigeonpeas',
                      'pomegranate','rice','watermelon']
        
        self.States = ['Camagüey', 'Ciego de Ávila', 'Cienfuegos', 'Granma', 'Guantánamo',
                    'Holguín', 'Isla de la Juventud', 'La Habana', 'Las Tunas','Matanzas', 
                    'Pinar del Río', 'Sancti Spíritus', 'Santiago de Cuba','Villa Clara']
        
        self.Rainfall_Models = ['Gradient Booosting Regressor', 'Linear Regression', 'Support Vector Regressor']

        # Humidity Standarization
        self.crop_df = pd.read_csv('../Datasets/Crop_recommendation.csv')
        self.humidity_Series = self.crop_df['humidity']
        self.humidity_mean = round(self.humidity_Series.mean(), 3)
        self.humidity_median = round(self.humidity_Series.median(), 3)

        # Crop Models Init
        self.DT_Crop = load('../Models/Crop/Crop_DT.joblib')
        self.GNB_Crop =  load('../Models/Crop/Crop_GNB.joblib')
        self.RF_Crop =  load('../Models/Crop/Crop_RF.joblib')
        self.SVM_Crop =  load('../Models/Crop/Crop_SVM.joblib')
        self.WC_Crop =  load('../Models/Crop/Crop_WC.joblib')
        # List of Crop Models
        self.crop_models_list = [self.DT_Crop, self.GNB_Crop, self.RF_Crop, self.SVM_Crop, self.WC_Crop]
        self.crop_models_names = ['DecissionTree', 'GausianNaivyBayes', 'RandomForest', 'SupportVectorMachine', 'WeigthedClassifier']

        # Weather Models Init
        ## Temperature
        self.LR_Temp = load('../Models/Weather/Temp_LR.joblib')

        ## Rainfall
        self.GBR_Rain = load('../Models/Weather/Rainfall_GBR.joblib')
        self.KR_Rain = load('../Models/Weather/Rainfall_KR.joblib')
        self.LR_Rain = load('../Models/Weather/Rainfall_LR.joblib')
        self.SVR_Rain = load('../Models/Weather/Rainfall_SVR.joblib')
        # List of Crop Models
        self.rain_model_list = [self.GBR_Rain, self.LR_Rain, self.SVR_Rain]

    
    def Crop_Prediction(self,X, model = 4):
        """
         Predict the crop label for the data. Crop_Label is a list of strings that correspond to the class labels in the crop_models_list
         
         @param X - pandas. DataFrame of shape [ n_samples n_features ]
         @param model - int default = 4 ( 4 ) model to use
         
         @return string of shape [ n_samples ] or None if model is not set to 4 or if prediction cannot be
        """
        prediction = self.crop_models_list[model].predict(X)
        return self.Crop_Label[prediction[0]]
    
    
    def Crop_Prediction_Each_Clf(self,X):
        """
         This function takes a list of data points and returns the prediction from each classifier. The classifiers are passed as arguments
         
         @param X - list of data points to be used for prediction
         
         @return list of labels for Crop_Predict_Each_Clf ( X ) where X is a
        """
        
        # Getting Prediction From Each Classifier
        predictions = []
        # Add predictions to predictions list
        for item in self.crop_models_list:
            pred = item.predict(X)
            predictions.append(pred[0])
        
        # Getting label
        predict_label = []
        # Add the label to the predict label.
        for item in predictions:
            predict_label.append(self.Crop_Label[item])
        
        return predict_label
    
    def Temp_Prediction(self,X):
        """
         Predict the temperature for each sample. This is a wrapper around LR_Temp. predict ( X )
         
         @param X - pandas. DataFrame of shape [ n_samples n_features ]
         
         @return prediction [ n_samples ] = Predicted temperatures for each sample in X. The mean is taken to be the same as the mean
        """
        prediction = self.LR_Temp.predict(X)
        return prediction
    
    def Rain_Prediction(self,X, model = 2):
        """
         Predict using Rain model. This is a wrapper for the predict method of the rain_model_list dictionary
         
         @param X - pandas. DataFrame of shape [ n_samples n_features ]
         @param model - int default = 2 ( 2 ) type of model to use
         
         @return numpy. ndarray of shape [ n_samples
        """
        prediction = self.rain_model_list[model].predict(X)
        return prediction
    
    def Normalize_Weather_Models_Input(self, State, Year):
        """
         This function takes the state and year of the weather model and normalizes it to 1x2.
         
         @param State - State of the weather model. Must be a float between 0 and 1.
         @param Year - Year of the weather model. Must be a float between 0 and 1.
         
         @return Input to the model as a numpy array of shape [ 1 x 2 ] where x is the state and year
        """
        input = [State, Year]
        x = np.array(input).reshape(1,2)
        return x
    
    def Normalize_Crop_Models_Input(self, temp, rain, humidity, ph, N, P, K):
        """
         This function takes the input from Crop_Models and normalizes it to the units of Telescope
         
         @param temp - Temperature in degrees Celsius.
         @param rain - Ravelle in degrees Celsius.
         @param humidity - Humidity in percent of centrifugal dilution.
         @param ph - Photon depolaris on the ground state.
         @param N - Number of nuclei in the government.
         @param P - Pressure in kgovernment.
         @param K - Number of crop models. A value of 0 corresponds to no crop models.
         
         @return A 1D array of shape ( N 7 ) where N is the number of nuclei in the government
        """
        input = [temp, rain, humidity, ph, N, P, K]
        x = np.array(input).reshape(1,7)
        return x
    
    def Future_Crop_Prediction(self, State, Year, humidity, ph, N, P, K, MultiPrediction=True):
        """
         This function is used in the Crop Prediction phase. It takes the state year humidity N P K prediction results and returns the crop prediction results
         
         @param State - The state of the forecast
         @param Year - The year of the forecast 
         @param humidity - The humidity
         @param ph - The phosphoryths of the forecast
         @param N - The number of predictors in the forecast ( N )
         @param P - The number of predictors in the forecast ( P )
         @param K - The number of predictors in the forecast ( K )
         @param MultiPrediction - Whether or not to use multi - prediction
         
         @return A list of crop prediction results
        """
        # Humidity values are 0 mean 1 median
        if humidity == 0:
            humidity = self.humidity_mean
        elif humidity == 1:
            humidity =self.humidity_median
        else:
            humidity = humidity

        temp = self.Temp_Prediction(self.Normalize_Weather_Models_Input(State, Year))
        rain = self.Rain_Prediction(self.Normalize_Weather_Models_Input(State, Year))
        
        
        if MultiPrediction: 
            crop = self.Crop_Prediction_Each_Clf(self.Normalize_Crop_Models_Input(temp[0], rain[0], humidity,ph, N, P, K))
        else:
            crop = self.Crop_Prediction(self.Normalize_Crop_Models_Input(temp[0], rain[0], humidity,ph, N, P, K)) 
        return crop

