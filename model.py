import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class SymptomChecker:
    def __init__(self, symptoms_path, precautions_path=None):
        self.df = pd.read_csv(symptoms_path)
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.prepare_data()
        self.train_model()

        # Load precautions data if provided
        if precautions_path:
            self.precautions = pd.read_csv(precautions_path)
        else:
            self.precautions = None

    def prepare_data(self):
        symptom_cols = [col for col in self.df.columns if 'Symptom' in col]
        self.df[symptom_cols] = self.df[symptom_cols].fillna('')
        self.df['combined_symptoms'] = self.df[symptom_cols].agg(' '.join, axis=1)
        self.X = self.vectorizer.fit_transform(self.df['combined_symptoms'])
        self.y = self.df['Disease']

    def train_model(self):
        self.model.fit(self.X, self.y)

    def predict(self, user_input):
        input_vector = self.vectorizer.transform([user_input])
        prediction = self.model.predict(input_vector)
        return prediction[0]

    def get_precautions(self, disease):
        if self.precautions is not None:
            row = self.precautions[self.precautions["Disease"] == disease]
            if not row.empty:
                return row.iloc[0, 1:].dropna().tolist()
        return []
