
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Example training dataset
# 0 = Low/No, 1 = High/Yes
# Temperature is in Celsius

data = {
    'Light_Level': [0, 1, 1, 0, 1, 0],
    'Motion': [1, 1, 0, 0, 1, 0],
    'Temperature': [30, 22, 24, 35, 26, 28],
    'Fan_On': [1, 0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

# Features and target
X = df[['Light_Level', 'Motion', 'Temperature']]
y = df['Fan_On']

# Training the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# Predicting fan status with new sensor input
prediction = model.predict([[1, 1, 27]])  # Predict Fan state
print("Prediction (Fan On?):", prediction[0])

# Visualize the decision tree
plt.figure(figsize=(10,6))
tree.plot_tree(model, feature_names=['Light_Level', 'Motion', 'Temperature'],
               class_names=['Off', 'On'], filled=True)
plt.title("Smart Home Decision Tree")
plt.show()
