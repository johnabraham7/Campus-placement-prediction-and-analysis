from sklearn.ensemble import RandomForestClassifier
import pickle
from utils.preprocess import preprocess_data

def train_and_save_model(data_filepath, model_filepath):
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data_filepath)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model and scaler
    with open(model_filepath, 'wb') as model_file:
        pickle.dump((model, scaler), model_file)

if __name__ == "__main__":
    # Update the file path to point to the CSV file in your project directory
    train_and_save_model('students_data.csv', 'models/placement_model.pkl')
