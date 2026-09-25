import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


np.random.seed(42)
n_samples = 2000

print("Generating synthetic logistics dataset...")
data = pd.DataFrame({
    'Distance_km': np.random.uniform(2, 50, n_samples),
    'Traffic_Index': np.random.randint(1, 11, n_samples),
    'Cargo_Weight_kg': np.random.uniform(10, 500, n_samples),
    'Driver_Experience_yrs': np.random.uniform(0.5, 15, n_samples),
    'Weather_Severity': np.random.choice([1, 2, 3], n_samples, p=[0.7, 0.2, 0.1])
})


data['Delivery_Time_Minutes'] = (data['Distance_km'] * 2.5) + (data['Traffic_Index'] * 3.2) * data['Weather_Severity'] - (data['Driver_Experience_yrs'] * 1.5) + np.random.normal(10, 5, n_samples)


X = data.drop('Delivery_Time_Minutes', axis=1)
y = data['Delivery_Time_Minutes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Regressor with GridSearchCV...")
rf = RandomForestRegressor(random_state=42)
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_




predictions = best_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n--- Model Evaluation Metrics ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Absolute Error (MAE): {mae:.2f} minutes")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} minutes")
print(f"R-squared (R²): {r2:.4f}")


importances = best_model.feature_importances_
feature_names = X.columns
print("\n--- Feature Importance (Optimization Drivers) ---")
for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance:.4f}")