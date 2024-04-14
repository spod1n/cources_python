import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class PenguinClassifier:
    def __init__(self):
        self.penguins = sns.load_dataset('penguins')
        self.penguins.to_csv('penguins_data.csv')
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        """
        Build the machine learning pipeline.
        Preprocesses the data, sets up feature transformations, and defines the classifier.
        """
        self.penguins.dropna(inplace=True)  # Drop rows with missing values

        # Separate features and target variable
        x = self.penguins.drop('species', axis=1)
        y = self.penguins['species']

        # Split data into training and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Pre-processing pipeline for numeric features
        numeric_features = x.select_dtypes(include=['number']).columns.tolist()
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

        # Pre-processing pipeline for categorical features
        categorical_features = x.select_dtypes(include=['object']).columns.tolist()
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])

        # Combine pre-processing pipelines
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
                                                       ('cat', categorical_transformer, categorical_features)])

        # Create the full pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', RandomForestClassifier())])
        return pipeline

    def train(self) -> None:
        """ Train the machine learning model on the training data """
        self.pipeline.fit(self.x_train, self.y_train)

    def validate(self) -> None:
        """ Validate the machine learning model using cross-validation """
        cv_scores = cross_val_score(self.pipeline, self.x_train, self.y_train, cv=5)
        print(f'Cross-validation scores: {cv_scores}')
        print(f'Mean cross-validation score: {cv_scores.mean()}')

    def fine_tune(self) -> None:
        """ Fine-tune the machine learning model using GridSearchCV """
        # Define the parameter grid for GridSearchCV
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }

        # Initialize GridSearchCV with the pipeline and parameter grid
        grid_search = GridSearchCV(self.pipeline, param_grid, cv=5)
        # Perform the grid search to find the best parameters
        grid_search.fit(self.x_train, self.y_train)

        # Get the best parameters found by GridSearchCV
        best_params = grid_search.best_params_
        print(f'Best parameters: {best_params}')

        # Update the pipeline with the best parameters and
        # retrain the model with the best parameters on the training data
        self.pipeline.set_params(**best_params)
        self.pipeline.fit(self.x_train, self.y_train)

    def evaluate_model(self) -> None:
        """ Evaluate the machine learning model on the test data and print performance metrics """
        y_pred = self.pipeline.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print('Accuracy:', accuracy, sep=' ', end='\n\n')
        print('Classification Report:', classification_report(self.y_test, y_pred), sep='\n', end='\n\n')
        print('Confusion Matrix:', confusion_matrix(self.y_test, y_pred), sep='\n', end='\n\n')

    def predict(self, new_data):
        """
        Make predictions on new data using the trained model.
        Args: new_data (DataFrame): New data to make predictions on.
        Returns: ndarray: Predicted labels for the new data.
        """
        return self.pipeline.predict(new_data)


if __name__ == '__main__':
    # Create and run the classifier
    classifier = PenguinClassifier()

    classifier.train()
    classifier.validate()
    classifier.fine_tune()
    classifier.evaluate_model()

    # Example usage of predict method
    new_data = classifier.x_test[:5]  # Example: Take the first 5 rows of test data
    predictions = classifier.predict(new_data)
    print('Predictions for new data:', predictions, sep='\n', end='\n\n')

"""
Модель досягла точності 1.0 (абсолютної точності) під час оцінки на тестовому наборі даних, 
що означає, що вона правильно класифікувала всі тестові зразки.

Оцінки precision, recall і F1-score також дорівнюють 1.0 для кожного класу, 
що свідчить про високу якість прогнозування для кожного виду пінгвіна.

Матриця помилок показує, що всі зразки класифікувалися правильно без помилок.

Модель дуже ефективна у прогнозуванні виду пінгвіна на основі фізичних характеристик та місцезнаходженням.
Вона добре узгоджується з навчальними даними (Mean Cross-validation scores = 0.985), що дає підстави для 
її використання в реальних задачах.
"""
