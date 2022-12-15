from datetime import datetime, timedelta
from tkinter import *
from tkinter import messagebox
from tkcalendar import Calendar
from tkinter import ttk

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error




tkobj = Tk()
# setting up the geomentry
tkobj.geometry("400x400")
tkobj.title("Weather Predictor")
current_time = datetime.now()

tkc = Calendar(tkobj,selectmode = "day",year=current_time.year,month=current_time.month,date=current_time.day)
#display on main window
tkc.pack(pady=40)


#function to predict and display data
def fetch_data():
    p['value'] = 0
    
    p.place(x=150, y=250)
    if p['value'] < 100:
                p['value'] += 4
                
                p.step()            
                tkobj.update()
    # if p['value'] < 100:
    # p['value'] = 25
    #clear existing items in tree
    for item in tree.get_children():
      tree.delete(item)
    #get date selected in calendar
    input_date =str(tkc.selection_get())
    print(input_date)

    if input_date == 'None':
        messagebox.showwarning("Warning", "Please Select Date")
    
    input_date = datetime.strptime(input_date, "%Y-%m-%d") 
    input_date -= timedelta(days=1)
    Begindate = input_date
    idatearr = []
   
    # p['value'] = 35
    #to get next 6 days date from selected date
    for i in range(7):

        Begindate += timedelta(days=1)
        strbegindate = str(Begindate)
        idatearr.append(strbegindate[0:10])
    

    #os.chdir("D:/")
    os.getcwd()
    creation=pd.read_csv('weather_dataset.csv')
    creation.head(5)


    creation_1 = creation.loc[creation['Temperature'] == 0.0]

    creation = creation.drop([x for x in creation_1.index])
    
    creation.describe()
    #convert cateogarical data to neumerical data
    creation=pd.get_dummies(creation)

    #to get the required column of the dataset to process data
    creation.iloc[:,5:].head(5)
    labels_1=np.array(creation['Temperature'])

   
    creation_1=creation.drop(['Hour','Minute','Wind Speed','Wind Direction','Temperature','Total Precipitation'],axis=1)

    creation_1=np.array(creation_1)

    

    
   #split data for training and testing purpose
    train_creation_1, test_creation_1, train_labels_1, test_labels_1= train_test_split(creation_1,labels_1, test_size=0.35,random_state=0)

    
    print('Training creation shape:', train_creation_1.shape)
    print('Training labels shape:', train_labels_1.shape)

    print('Testing creation shape:', test_creation_1.shape)
    print('Testing label shape:', test_labels_1.shape)
    # p['value'] = 65
    #random forest regressor model to predict values
    rf_1=RandomForestRegressor(n_estimators=1000, random_state=0)
    rf_1.fit(train_creation_1, train_labels_1)
    #prediction on test data
    predictions_1=rf_1.predict(test_creation_1)
    errors=abs(predictions_1 - test_labels_1)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    mape=100*(errors/test_labels_1)
    accuracy=100-np.mean(mape/3)
    print('Accuracy of the model:', round(accuracy,2),'%')
    

    #adding columns in tree view

    tree.column("# 1", anchor=CENTER)
    tree.heading("# 1", text="Date")
    tree.column("# 2", anchor=CENTER)
    tree.heading("# 2", text="Temprature")
    tree.column("# 3", anchor=CENTER, width = 250)
    tree.heading("# 3", text="alerts")
    
    #add  date wise data in tree view
    for dt in idatearr:

        tree.pack()
       
        date_val = datetime.strptime(dt, "%Y-%m-%d")
        input_1 = [[date_val.year,date_val.day,date_val.month]]
        predictions_1=rf_1.predict(input_1)
        date_time = str(date_val)[0:10]
        temprature = round(predictions_1[0],2)

        p['value'] = 100
        
        #updating alert value as per the condition
        if temprature<6:
            alert = "Too Cold - Not convinient to move out"
            tree.insert('', 'end', text="1", values=(str(date_time), str(temprature), str(alert)))
        elif temprature>45:
            alert = "Too Hot - Not convinient to move out"
            tree.insert('', 'end', text="1", values=(str(date_time), str(temprature), str(alert)))
        else:
            alert = "None"
            tree.insert('', 'end', text="1", values=(str(date_time), str(temprature), str(alert)))
        
        
#LinearRegression Model to predict data

#     reg = LinearRegression()         
#     model_4= reg.fit(train_creation_1, train_labels_1)

#     # Predict
#     y_pred_train = model_4.predict(train_creation_1)
#     y_pred_test = model_4.predict(test_creation_1)

#     # model evaluation
#     mse_train = mean_squared_error(train_labels_1, y_pred_train)
#     mse_test = mean_squared_error(test_labels_1, y_pred_test)

#     rmse_train = np.sqrt(mean_squared_error(train_labels_1, y_pred_train))
#     rmse_test = np.sqrt(mean_squared_error(test_labels_1, y_pred_test))

#     mae_train = mean_absolute_error(train_labels_1, y_pred_train)
#     mae_test = mean_absolute_error(test_labels_1, y_pred_test)
#     r2 =model_4.score(train_creation_1, train_labels_1)


    
#     # printing values
#     # print('Slope:' ,model.coef_)
#     # print('Intercept:', model.intercept_)

#     print("#### observations for model 4 #####")
#     print('mean squared error for train : ', mse_train)
#     print('mean squared error for validation: ', mse_test)
#     print('root mean squared error for train: ', rmse_train)
#     print('root mean squared error validation: ', rmse_test)
#     print('mean absolute error for train: ', mae_train)
#     print('mean absolute error for validation: ', mae_test)
#     print('mean absolute percentage error for train: ', mean_absolute_percentage_error(train_labels_1, y_pred_train))
#     print('mean absolute percentage error for validation: ', mean_absolute_percentage_error(test_labels_1, y_pred_test))
#     print('R2 score: ', r2)   

# # function to calulate MAPE
# def mean_absolute_percentage_error(y_true, y_pred): 
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
#add button to load the date clicked on calendar
but = Button(tkobj,text="Predict Weather Report",command=fetch_data, bg="black", fg='white')
#displaying button on the main display
but.pack()
tree = ttk.Treeview(tkobj, column=("Date", "Temprature", "alerts"), show='headings', height=7)
p = ttk.Progressbar(tkobj,orient=HORIZONTAL,length=200,mode="determinate",takefocus=True)
#starting the object
tkobj.mainloop()