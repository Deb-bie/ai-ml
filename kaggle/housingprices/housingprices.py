import pandas as pd

melb_file_path = 'D://My files//ML//kaggle//housingprices//melb_data.csv'
melb_data = pd.read_csv(melb_file_path)     # reading the data in the csv file 

print("size of the housing price data frame", melb_data.shape)
print(melb_data.describe()) # printing the data to see its shape and form

print(melb_data[0:5])

