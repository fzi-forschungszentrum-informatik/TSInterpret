from data.load_data import load_multivariate_data
from ClassificationModels.LSTM import model
from Visualizations.general_plot import data_resample_plot

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


#def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
#    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trainx, testx, trainy, testy,org=load_multivariate_data('./data/Multivariate/household_power_consumption.txt')
    data_resample_plot(org,columname='Sub_metering_1')
    model(trainx,trainy, testx,  testy)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
