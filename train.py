import pandas as pd
#Load Dataset
data = pd.read_csv('house_data.csv')
print(data.head())

#Explore Data
data.info()
data.isnull().sum()
data.describe()

#Visualize
import matplotlib.pyplot as plt
plt.scatter(data['Area'], data['Price'])
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

#Preprocess Data
X = data[['Area', 'Bedrooms', 'Bathrooms', 'Age', 'Location_Score']]
y = data['Price']

#Train Split Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train AI Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

#Make Prediction
predictions = model.predict(X_test)
print(predictions[:5])

#Evaluate Model
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predictions)
print(mae)
