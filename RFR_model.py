# Importing the required Python Libraries to handle the data
import pandas as pd
import pickle

# Import the datasets
for fileName in ['Stock1_AMC', 'Stock2_BAC', 'Stock3_NVDA']:
    data = pd.read_csv(fileName+'.csv')
    print()
    print(fileName)
    print()
    
    # Separting Features from the Target
    X = data[['Date', 'Open','High', 'Low', 'Close', 'Volume']]
    Y = data['Adj Close']
    
    
# Splitting data into intial training and validation sets from the original data with a ratio of 8:2 
# i.e. 80% and 20% respectively.
    from sklearn.model_selection import train_test_split
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, random_state=101)
    
    # Splitting validation data further into validation:testing sets with ratio of 50:50
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_valid, Y_valid, train_size=0.5, random_state=101)
    
# P.S. Note that 0.5*0.2 = 0.1 so the final effect of these two splits is to have the original data split  
# into training/validation/test sets in 8:1:1 ratio i.e. 80%, 10% and 10% respectively.
    
    print("len(X): {} len(Y): {} \nlen(X_train): {}, len(X_valid): {}, len(X_test): {}\
    \nlen(Y_train): {}, len(Y_valid): {}, len(Y_test): {}".format(len(X), len(Y),\
    len(X_train), len(X_valid), len(X_test), len(Y_train), len(Y_valid), \
    len(Y_test))) 
    
    #print(X_valid)
    validation_data= pd.concat([X_valid, Y_valid], axis=1, join="inner")
    validation_data.to_csv(fileName+'_valid.csv')
    
    X_train = X_train [['Open', 'High', 'Low', 'Close', 'Volume']]
    X_valid = X_valid [['Open', 'High', 'Low', 'Close', 'Volume']]
    X_test = X_test [['Open', 'High', 'Low', 'Close', 'Volume']]
    
# Data Normalization using MinMax Scaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))     
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test) 
    
# Save the mean and variance of each stock calculated using MinMax scaler during fitting with training features
# as valid scaled pickle files to the disk to load it later in GUI as use for prediction
    pickle.dump(scaler, open(fileName+'_valid_scaled.pickle', 'wb'))
    
    """ Random Forest Regressor (RFR) Algorithm """
    
    # Importing the Algorithm from the library
    from  sklearn.ensemble import RandomForestRegressor
    rfr = RandomForestRegressor()
    
    # Fitting the model to the training dataset
    rfr.fit(X_train, Y_train)
    
    print()
    print('RFR Training Score for '+fileName+':',rfr.score(X_train,Y_train))
    print('RFR Validation Score for '+fileName+':',rfr.score(X_valid,Y_valid))
    print('RFR Testing Score for '+fileName+':',rfr.score(X_test,Y_test))
    print()
    
# Predicting the Test Results
    Y_pred_rfr = rfr.predict(X_test)
    
# Calcukate the three metrics for Random Forest Regressor
    
# Importing the required metrics to calculate statistics
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import r2_score as r2
    
# Calculate the Mean Absolute Error 
    rfr_mae = mae(Y_test, Y_pred_rfr)
    print('RFR MAE Score for ' +fileName+':', rfr_mae)
    
# Calculate the Mean Square Error
    rfr_mse = mse(Y_test, Y_pred_rfr)
    print('RFR MSE Score for ' +fileName+':', rfr_mse)
    
# Calculate the r2 Score
    rfr_r2 = r2(Y_test, Y_pred_rfr)
    print('RFR R2 Score for ' +fileName+':', rfr_r2)
    print()
    
# Save the RFR models to the disk
    pickle.dump(rfr, open(fileName+'_valid_model.pickle', 'wb'))
    
