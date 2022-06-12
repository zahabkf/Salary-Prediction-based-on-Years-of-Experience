import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model, model_selection
import os.path

# Importing the data set
data = pd.read_csv("Salary_Data.csv", sep=',')
predict = 'Salary'
parameter= 'YearsExperience'

X = np.array(data[parameter])
Y = np.array(data[predict])
X = X.reshape((30,1))
Y= Y.reshape((30,1))

# checking if the model already exists or not
# if it doesn't, then create a new model which has above 98%  accuracy
# otherwise load the pre-existing model
if os.path.exists("salarymodel.pickle"):
    print("Loading Model...")
    model = pickle.load(open("salarymodel.pickle","rb"))
else:
    print("Creating a new Model...")
    best_accuracy = 0
    model_score = 0

    while model_score < 0.98:

        # Splitting the data set into train set and test set
        x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y,test_size= 0.2)

        # Fitting simple Linear Regression to the train set
        model = linear_model.LinearRegression()
        model = model.fit(x_train, y_train)

        # Checking the model accuracy
        model_score = model.score(x_test,y_test)
        print(model_score)
    # Saving the model
    with open("salarymodel.pickle","wb") as file:
        pickle.dump(model,file)

# prompting user to input years of experience to predict the salary
years_of_exp = np.array([[float(input("Enter the years of experience: "))]])

# Predicting the salary based on the years of experience
print(model.predict(years_of_exp))