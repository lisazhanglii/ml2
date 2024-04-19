import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from dagshub import DAGsHubLogger
import hashlib
import dagshub
import os

dagshub.init(repo_owner='yixinzha', repo_name='assign2', mlflow=True)


# Load the dataset
file_path = 'vineyard_weather_1948-2017.csv'
data = pd.read_csv(file_path)

# Convert 'DATE' to datetime format
data['DATE'] = pd.to_datetime(data['DATE'])

# Extract week number and year
data['YEAR'] = data['DATE'].dt.year
data['WEEK'] = data['DATE'].dt.isocalendar().week

# Filter for weeks 35 to 40
filtered_data = data[(data['WEEK'] >= 35) & (data['WEEK'] <= 40)]

# Aggregate data by year and week
weekly_aggregated_data = filtered_data.groupby(['YEAR', 'WEEK']).agg(
    AVG_PRCP=('PRCP', 'mean'),
    MAX_TMAX=('TMAX', 'max'),
    MIN_TMIN=('TMIN', 'min'),
    DAYS_RAIN=('RAIN', 'sum')
).reset_index()

# Creating lag features
weekly_aggregated_data['AVG_PRCP_t-1'] = weekly_aggregated_data.groupby(['YEAR'])['AVG_PRCP'].shift(1)
weekly_aggregated_data['DAYS_RAIN_t-1'] = weekly_aggregated_data.groupby(['YEAR'])['DAYS_RAIN'].shift(1)
weekly_aggregated_data['AVG_PRCP_t-2'] = weekly_aggregated_data.groupby(['YEAR'])['AVG_PRCP'].shift(2)
weekly_aggregated_data['DAYS_RAIN_t-2'] = weekly_aggregated_data.groupby(['YEAR'])['DAYS_RAIN'].shift(2)

# Generate 'STORM_NEXT_WEEK' based on 'DAYS_RAIN' (adjust logic as needed)
weekly_aggregated_data['STORM_NEXT_WEEK'] = (weekly_aggregated_data['DAYS_RAIN'] > 3).astype(int)
weekly_aggregated_data['STORM_NEXT_WEEK'] = weekly_aggregated_data.groupby(['YEAR'])['STORM_NEXT_WEEK'].shift(-1)

# Drop rows with NaN values resulting from shift operation or initial lag feature creation
cleaned_data = weekly_aggregated_data.dropna().reset_index(drop=True)

# Prepare features and target variable
X = cleaned_data.drop(['YEAR', 'WEEK', 'STORM_NEXT_WEEK'], axis=1)  # Exclude 'YEAR' and 'WEEK' from features
y = cleaned_data['STORM_NEXT_WEEK'].astype(int)  # Target variable

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from dagshub import DAGsHubLogger

from dagshub import DAGsHubLogger

def run_experiment(params, experiment_name):
    # Initialize the classifier
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred)
    }

    # Initialize the DAGsHub logger
    logger = DAGsHubLogger()

    # Log parameters and metrics
    logger.log_hyperparams(params)
    logger.log_metrics(metrics)

    # Don't forget to close the logger
    logger.close()

    return metrics



# Ensure the directories for logging exist
os.makedirs('./metrics', exist_ok=True)
os.makedirs('./params', exist_ok=True)

# Define different sets of parameters for each experiment
param_sets = [
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 150, 'max_depth': 15},
    {'n_estimators': 200, 'max_depth': 20}
]

# Run experiments
for i, params in enumerate(param_sets, start=1):
    experiment_name = f'experiment_{i}'
    metrics = run_experiment(params, experiment_name)
    print(f"Metrics for {experiment_name}: {metrics}")