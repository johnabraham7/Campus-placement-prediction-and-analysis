import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(filepath):
    data = pd.read_csv(filepath)

    # Encode the target variable 'Placement Outcome'
    data['Placement Outcome'] = data['Placement Outcome'].apply(lambda x: 1 if x == 'Placed' else 0)

    # Separate features and target
    features = data.drop('Placement Outcome', axis=1)
    target = data['Placement Outcome']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler
