# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results. 

## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.array([[5.0], [5.5], [6.0], [6.5], [7.0], [7.5], [8.0], [8.5]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])


model = LogisticRegression()
model.fit(X, y)

cgpa = [[6.8]]
result = model.predict(cgpa)

print("Prediction (0 = Not Placed, 1 = Placed):", result)

plt.scatter(X, y, color='red', label='Actual Data')

X_test = np.linspace(4.5, 9.0, 100).reshape(-1, 1)
plt.plot(X_test, model.predict_proba(X_test)[:, 1],
         color='blue', label='Placement Probability Curve')

plt.xlabel("CGPA")
plt.ylabel("Probability of Placement")
plt.title("Logistic Regression: Student Placement Prediction")
plt.legend()
plt.grid(True)
plt.show()


Developed by: Madhu Shruthi A.R
RegisterNumber:  25008368
```

## Output:
![WhatsApp Image 2026-01-30 at 2 38 12 PM](https://github.com/user-attachments/assets/f8cabdf7-5258-4f49-a1cc-bb5c09e78fcc)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
