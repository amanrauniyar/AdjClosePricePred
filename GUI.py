# Import the required python libraries for the GUI
import pandas as pd
import tkinter as tk
from tkinter import ttk
import pickle

# Creating the base window for the Tkinter GUI
window = tk.Tk()
window.title("Aman's Stock Price Prediction")
window.geometry("680x400")

# Initialize selected_stock, selected_date, data, (open, high, low, close, adjusted closing) price, 
# volume, loaded_model and scaler in global scope

openPrice = 0
high = 0
low = 0
close = 0
adj_close = 0
volume = 0
data = None
loaded_model = None
scaler = None

# Creating the title label for the GUI
titleLabel = tk.Label(window, text="Prediction of Stock Value using Random Forest Regression Algorithm"
                , font = ('Book Antiqua', 16, 'bold'))
# Sticking it onto the screen
titleLabel.grid()

# Creating a label for the description of the application
subtitleLabel = tk.Label(window, text="\n Trained with RFR model and loaded with the validation data for prediction"
                         , font = ('Comic Sans MS', 14), fg="orange")
# Sticking it onto the screen
subtitleLabel.grid(row=2, padx=5, sticky='w')

""" Here onwards, used place to stick the label and widget on screen 
since, grid was being harder to adjust the positions precisely """

# Creating a label for the stocks
stocksLabel = tk.Label(window, text="\n Stocks:", font = ('Cascadia Code', 13), fg="purple")
# Sticking it onto the screen
stocksLabel.place(x=9, y=85)

# Creating a dropdown widget for the list of stocks
# Use a string var to store the value currently selected from the combobox
selected_stock = tk.StringVar() 
# Set the string var as the text variable in the combobox
stocks_cb = ttk.Combobox(window, textvariable = selected_stock)    
# Set the values in your stock selection combobox, in this implementation they are the names of the 
# csv files without the .csv extension, we will add that later
stocks_cb['values'] = ["Stock1_AMC_valid", "Stock2_BAC_valid", "Stock3_NVDA_valid"] 
# Sticking it onto the screen
# stocks_cb.grid(row=4, padx=14, pady=2, sticky='w')
stocks_cb.place(x=21, y=130)

# Creating a label for the dates
dateLabel = tk.Label(window, text="\n Dates: ", font = ('Cascadia Code', 13), fg="purple")
# Sticking it onto the screen
# dateLabel.grid(row=5, pady=2, sticky='w')
dateLabel.place(x=9, y=160)

# Creating a dropdown widget for the list of dates
# Use a string var to store the value currently selected from the combobox
selected_date = tk.StringVar()
# Set the string var as the text variable in the combobox
dates_cb = ttk.Combobox(window, textvariable = selected_date)
# Sticking it onto the screen
# dates_cb.grid(row=6, padx=14, pady=2, sticky='w')
dates_cb.place(x=21, y=205)

# Creating a widget and a label for all the different prices and volume feature

""" Open Price Label """
opLabel = tk.Label(window, text="\n\t\t\t        Opening Price: ", font = ('Times New Roman', 13) )
# Sticking it onto the screen
# opLabel.grid(row=4, sticky='e')
opLabel.place(x=200, y=130)

""" High Price Label """
hpLabel = tk.Label(window, text="\t\t\t              High Price: ", font = ('Times New Roman', 13) )
# Sticking it onto the screen
# hpLabel.grid(row=5)
hpLabel.place(x=200, y=180)

""" Low Price Label """
lpLabel = tk.Label(window, text="\t\t\t               Low Price: ", font = ('Times New Roman', 13) )
# Sticking it onto the screen
# lpLabel.grid(row=6)
lpLabel.place(x=200, y=210)

""" Closing Price Label """
cpLabel = tk.Label(window, text="\t\t\t         Closing Price: ", font = ('Times New Roman', 13) )
# Sticking it onto the screen
# cpLabel.grid(row=7)
cpLabel.place(x=200, y=240)

""" Adjusted Closing Price Label """
acpLabel = tk.Label(window, text="\t\t           Adjusted Closing Price: ", font = ('Times New Roman', 13) )
# Sticking it onto the screen
# acpLabel.grid(row=8)
acpLabel.place(x=200, y=270)

""" Volume Label """
vLabel = tk.Label(window, text="\t\t\t\t Volume: ", font = ('Times New Roman', 13) )
# Sticking it onto the screen
# vLabel.grid(row=3, sticky='e')
vLabel.place(x=200, y=120)

""" Final Prediction Label fpr Adjusted Closing Price """

# # Creating an entry box to display the predicted adjusted closing price of the stock for the current day
# ebpacvLabel = tk.Entry(window, font = ('Helvetica', 14), width=13, fg="blue", bg="yellow", bd=0) 
# ebpacvLabel.grid(row=9, column=0, pady = 5)

# Creating a widget and label for the predicted adjusted closing price of the stock for the current day
pacpLabel = tk.Label(window, text="\n Predicted Adjusted Closing Price for the Current Day:"
                  , font = ('Berlin Sans FB', 15))
# pacpLabel.grid(row=9, pady=11)
pacpLabel.place(x=107, y=292)

# This function will be bound to the stock selection combobox
# When a stock is selected, the appropriate csv file will be loaded and we will pull out the dates and set the date combobox
def stock_changed(event):
# Use the global data object    
    global data, loaded_model, scaler
# Get the value from the string var that stores the current stock selected from the combobox
    s = selected_stock.get()
# Load the csv file using that value, add the .csv extension back, store it in the global data object so we can use it elsewhere    
    data = pd.read_csv(f'./{s}.csv')   
# Pull out the column of dates and set it as the dates combox values
    dates_cb['values'] = list(data['Date'])  
    print(s)
    # Load the saved RF Regression model saved as pickle from disk
    loaded_model = pickle.load(open(f'./{s}_model.pickle', 'rb'))
    # Load the scaled data from the saved pickle file from disk
    scaler = pickle.load(open(f'./{s}_scaled.pickle', 'rb'))
    

# This function will be bound to the date selection combobox
# When a date is selected we will get the row of data for that date and set the open, high, low, close values accordingly
def date_selected(event):  
    global data, openPrice, high, low, close, adj_close, volume # Load the global variables
    # Get the value from the string var that stores the current date selected from the combobox 
    d = selected_date.get() 
    print(d)
    if data is not None:    # Make sure the data variable is not None to avoid errors
        row = data.loc[data['Date'] == d]   # This fancy pandas code pulls out the row that has a date matching the date we have selected
        print(row)
        # Set our openPrice, high, low and close variables with the data from this row
        openPrice = float(row['Open'])
        high = float(row['High'])
        low = float(row['Low'])
        close = float(row['Close'])
        adj_close = float(row['Adj Close'])
        volume = int(row['Volume'])
        # Update our labels
        opLabel.config(text = f'\n\t\t\t        Opening Price: {openPrice:.2f}')
        hpLabel.config(text = f'\t\t\t              High Price: {high:.2f}')
        lpLabel.config(text = f'\t\t\t               Low Price: {low:.2f}')
        cpLabel.config(text = f'\t\t\t         Closing Price: {close:.2f}')
        acpLabel.config(text = f'\t\t           Adjusted Closing Price: {adj_close:.2f}')
        vLabel.config(text = f'\t\t\t\t Volume: {volume}')
        
# Bind the stock_changed function to the stock combobox
stocks_cb.bind('<<ComboboxSelected>>', stock_changed)    

# Bind the date_selected function to the date combobox
dates_cb.bind('<<ComboboxSelected>>', date_selected)

""" Predict Button for Adjusted Closing Price """
def predictClick():
    global openPrice, high, low, close, volume, loaded_model, scaled
# It will take (open, high, low, close) price and volume as inputs.
    inputs = [openPrice, high, low, close, volume]
# Now, it will transform those inputs with the help of loaded scaled data
    scaled_inputs = scaler.transform([inputs])
# It will predict the adjusted closing price based on the scaled inputs
    result = loaded_model.predict(scaled_inputs)
# Print the result on the python console
    print (result)
# Postioned the entry box for final prediction of closing price
    # ebpacvLabel.delete(0, 'end')
    # ebpacvLabel.insert(0, str(result[0]))
    
# Print the formatted result on the GUI
    pacpLabel.config(text="\n Predicted Adjusted Closing Price for the Current Day: {:.2f}".format(result[0]), fg="blue" )

# Create a button to predict the prce after you choose the stock and select the date
predictButton = tk.Button(window, text="Predict...", font=('Cascadia Code', 11), fg="green", command=predictClick)
# predictButton.grid(row=8, padx=14, sticky='w')
predictButton.place(x=25, y=255)

""" Reset Button """

def resetClick():
    global dates_cb
    selected_stock.set("")
    dates_cb['values'] = list()
    selected_date.set("")
    opLabel.config(text = "\n\t\t\t        Opening Price:")
    hpLabel.config(text = "\t\t\t              High Price:")
    lpLabel.config(text = "\t\t\t               Low Price:")
    cpLabel.config(text = "\t\t\t         Closing Price:")
    acpLabel.config(text = "\t\t           Adjusted Closing Price:")
    vLabel.config(text = "\t\t\t\t Volume:")
    pacpLabel.config(text = "\n Predicted Adjusted Closing Price for the Current Day:", fg="black")

# Create a button to reset the prices and volume
resetButton = tk.Button(window, text="Reset", font=('Cascadia Code', 11), fg="dark red", command=resetClick)
resetButton.place(x=600, y=360)

# Create a button to exit the application
quitButton = tk.Button(window, text="Quit", font=('Cascadia Code', 11), fg="red", command=window.destroy)
quitButton.place(x=25, y=360)

# Creating a main event loop
window.mainloop()


