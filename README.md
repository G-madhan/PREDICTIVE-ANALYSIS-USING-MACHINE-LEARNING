# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING

*COMPANY*: CODTECH IT SOLUTION

*NAME*: MADHAN G

*INTERN ID*: CT08VCD

*DOMAIN*: DATA ANALYTICS

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH

Predictive analysis is essentially the practice of using data, statistical algorithms, and machine learning techniques to identify the likelihood of future outcomes based on historical data.In simpler terms, it's about using past information to try and figure out what might happen next.

dataset: https://www.kaggle.com/datasets/farzadnekouei/gold-price-10-years-20132023/

**Import necessary libraries:** 
import pandas as pd
import numpy as np


**Load the dataset into a Pandas DataFrame: **
df = pd.read_csv('your_dataset.csv')  # Replace 'your_dataset.csv' with the actual filename

**Explore the data:** 
df.head()
df.info()
df.describe()

**Clean the data**: This might involve handling missing values, converting data types, and removing irrelevant columns.
**Feature engineering:** Create new features that might improve the model's performance. For example, you could calculate moving averages or other technical indicators.

1**.Day of the week**: Create a new feature representing the day of the week (Monday, Tuesday, etc.). Gold prices might exhibit weekly patterns due to market activity. 
		df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek

**Month:** Create a feature for the month. Seasonality could influence gold prices. 
		df['Month'] = pd.to_datetime(df['Date']).dt.month

**Week of the year**: Extract the week number to capture potential yearly trends.
		df['WeekOfYear'] = pd.to_datetime(df['Date']).dt.isocalendar().week
Holidays: If you have data spanning multiple years, you could create a binary feature indicating whether a particular date is a holiday. Holiday periods might have different price dynamics.

**2. Lagged Features:**
Previous day's price: Include the gold price from the previous day as a feature. This captures recent price trends.
 
		df['PreviousPrice'] = df['Price'].shift(1)


**Moving averages:** Calculate moving averages (e.g., 7-day, 30-day) of the price and other relevant features. Moving averages smooth out short-term fluctuations and highlight longer-term trends.
 
		df['MA7'] = df['Price'].rolling(window=7).mean()
		df['MA30'] = df['Price'].rolling(window=30).mean()

** Data Splitting**
Purpose: Divide your dataset into training and testing sets to evaluate the performance of your machine learning model. 

		from sklearn.model_selection import train_test_split
    		X = df,drop(“Price”,aaxis=1) 
   		 y = df['Price'] 
   		 X_train, X_test, y_train, y_test = train_test_split(X, y, t

test_size=0.2`: This allocates 20% of the data for testing and 80% for training. You can adjust this as needed.
`random_state=42`: This ensures reproducibility of the split.

**Create and train the model:**
 		from sklearn.linear_model import LinearRegression  
 		model = LinearRegression()
 		model.fit(X_train, y_train)

**Model Evaluation**
Make predictions on the test set:
		y_pred = model.predict(X_test)

Evaluate the model's performance using appropriate metrics:
**R-squared (R2):** Measures the proportion of variance in the target variable explained by the model. Higher values (closer to 1) indicate better performance.
**Mean Squared Error (MSE):** Measures the average squared difference between predicted and actual values. Lower values are better.
**Root Mean Squared Error (RMSE):** The square root of MSE, providing a more interpretable measure of error.
**Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual value.
 
		from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
  		r2 = r2_score(y_test, y_pred)
    		mse = mean_squared_error(y_test, y_pred)
    		rmse = np.sqrt(mse)
    		mae = mean_absolute_error(y_test, y_pred)
    		print(f"R-squared: {r2}")
   		print(f"MSE: {mse}")
   		print(f"RMSE: {rmse}")
   		print(f"MAE: {mae}")

**Prediction on New Data**
Once you're satisfied with the model's performance, use it to predict future gold prices:
		
new_date = pd.to_datetime('2022-12-31')
open=1821
new_data = pd.DataFrame({
    'DayOfWeek': [new_date.dayofweek],
    'Month': [new_date.month],
    'WeekOfYear': [new_date.isocalendar().week],
    'Open': [open]

})

predicted_price = model.predict(new_data)
print("Predicted Price:",np.round(predicted_price[0],2))

**SAMPLE DATASET**
![sample gold price data](https://github.com/user-attachments/assets/0c156099-543a-4711-b212-d1d6c4bac0c2)

**SAMPLE OUTPUT**
![SAMPLE OUTPUT](https://github.com/user-attachments/assets/f05fed32-3ad2-401c-844e-1154baa13f4f)




