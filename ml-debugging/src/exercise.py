from typing import List, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore


RANDOM_STATE = 42


def pre_process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the Titanic dataset before it can be used to train a model.
    """
    processed_df = df.copy()
    processed_df = processed_df.dropna(subset=["Age", "Sex", "Pclass", "Embarked"])
    processed_df["Sex"] = processed_df["Sex"].map({"female": 0, "male": 1})
    processed_df["Embarked"] = processed_df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    processed_df = processed_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    return processed_df


def calculate_metrics(expected: List[int], predicted: List[int]) -> Tuple[float, float, float]:
    """
    Calculate accuracy, precision and recall metrics of two datasets that contain binary data.
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

    accuracy = true_positives / (true_positives + false_positives)
    precision = true_positives / (true_positives + false_negatives)
    recall = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)

    return accuracy, precision, recall


def main() -> None:
    print("ADSP ML Debugging Exercise\n")

    # Load the Titanic dataset
    print("Loading the Titanic dataset...")
    titanic_data = pd.read_csv("train.csv")

    # Preprocess the data
    print("Preprocessing the data...")
    titanic_data = pre_process_data(titanic_data)

    # Select features and target
    target = "Survived"
    X = titanic_data[titanic_data.columns]
    y = titanic_data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train a Random Forest classifier
    print("Training a Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=2)

    model.fit(X_test, y_test)

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
    print("Making predictions on unseen data...")
    # TODO You should make predictions here against the unseen data from the test.csv file

    return


if __name__ == "__main__":
    main()
