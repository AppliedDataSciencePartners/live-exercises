from typing import List, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore


RANDOM_STATE = 42


def pre_process_data(df: pd.DataFrame, columns_to_keep: List) -> pd.DataFrame:
    """
    changed this method slightly to accommodate both the training and test set
    """
    processed_df = df.copy()
    processed_df = processed_df.dropna(subset=["Age", "Sex", "Pclass", "Embarked"])
    processed_df["Sex"] = processed_df["Sex"].map({"female": 0, "male": 1})
    processed_df["Embarked"] = processed_df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    # encoding Pclass
    processed_df['Pclass'] = processed_df['Pclass'].map({
        '1': 1,
        '1st': 1,
        '2': 2,
        '2nd': 2,
        '3': 3,
        '3rd': 3
    })

    processed_df = processed_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    # ensuring the features that we need are retained
    for feature in columns_to_keep:
        if feature not in columns_to_keep:
            processed_df[feature] = 0 # some default value that felt not too terrible
    
    # ensuring the ordering of the features is retained or else it messes the ouptut
    processed_df = processed_df[columns_to_keep]

    return processed_df


def calculate_metrics(expected: List[int], predicted: List[int]) -> Tuple[float, float, float]:
    """
    corrected the formulas for accuracy, precision and recall
    """
    true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0

    for i in range(len(expected)):
        if expected[i] == predicted[i]:
            if expected[i] == 1:
                true_positives += 1
            else:
                true_negatives += 1
        else:
            if expected[i] == 1:
                false_negatives += 1
            else:
                false_positives += 1

    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0 
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return accuracy, precision, recall


def main() -> None:
    print("ADSP ML Debugging Exercise\n")

    # Load the Titanic dataset
    print("Loading the Titanic dataset...")
    titanic_data = pd.read_csv(r"C:\Users\Abin\OneDrive\Desktop\ADSP\adsp_interview\live-exercises\ml-debugging\src\train.csv")

    # Select features and target
    target = "Survived"
    features = ['Pclass', 'Sex', 'Embarked', 'Age', 'Fare']

    # Preprocess the data
    print("Preprocessing the data...")
    titanic_data = pre_process_data(titanic_data, columns_to_keep = features + [target])

    X = titanic_data[features]
    y = titanic_data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Train a Random Forest classifier
    print("Training a Random Forest classifier...")
    model = RandomForestClassifier(n_estimators = 500)

    model.fit(X_train, y_train)

    # Make predictions on the test set
    print("Making predictions on the test set...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    print("Calculating metrics...\n")
    accuracy, precision, recall = calculate_metrics(y_test.tolist(), y_pred.tolist())
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    # Make predictions on unseen data
    print("Making predictions on unseen data by Abin Varghese...")
    # TODO You should make predictions here against the unseen data from the test.csv file
    test_data = pd.read_csv(r'C:\Users\Abin\OneDrive\Desktop\ADSP\adsp_interview\live-exercises\ml-debugging\src\test.csv')
    test_data = pre_process_data(test_data, columns_to_keep = features)

    test_predictions = model.predict(test_data)

    print(f"The predictions on the test data:\n{test_predictions}")


if __name__ == "__main__":
    main()