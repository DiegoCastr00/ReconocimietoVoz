from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC

class AudioEnsembleClassifier:
    def __init__(self):
        # Definir varios modelos base
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'lr': LogisticRegression(max_iter=1000, random_state=42),
            'svm': SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
        }
        
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.weights = None
        
    def add_noise(self, X, noise_factor=0.05):
        """Añade ruido gaussiano a los datos para aumentar la robustez."""
        noise = np.random.normal(0, noise_factor, X.shape)
        return X + noise
    
    def train(self, X, y, cv_folds=5):
        """Entrena el ensemble con validación cruzada para determinar pesos."""
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Escalar datos
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Aumentar datos con ruido
        X_train_noisy = self.add_noise(X_train_scaled)
        X_train_combined = np.vstack([X_train_scaled, X_train_noisy])
        y_train_combined = np.hstack([y_train, y_train])
        
        # Entrenar modelos y obtener scores
        cv_scores = {}
        for name, model in self.models.items():
            print(f"\nEntrenando {name}...")
            scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)
            cv_scores[name] = scores.mean()
            
            # Entrenar en todos los datos aumentados
            model.fit(X_train_combined, y_train_combined)
            self.trained_models[name] = model
        
        # Calcular pesos basados en los scores de validación cruzada
        total_score = sum(cv_scores.values())
        self.weights = {name: score/total_score for name, score in cv_scores.items()}
        
        # Evaluar ensemble
        y_pred = self.predict(X_test_scaled)
        
        print("\nReporte de clasificación del ensemble:")
        print(classification_report(y_test, y_pred))
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión del Ensemble')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.show()
        
        return X_test_scaled, y_test, y_pred
    
    def predict(self, X):
        """Realiza predicción usando votación ponderada."""
        X_scaled = self.scaler.transform(X)
        predictions = np.zeros((X_scaled.shape[0], len(np.unique(next(iter(self.trained_models.values())).classes_))))
        
        for name, model in self.trained_models.items():
            predictions += self.weights[name] * model.predict_proba(X_scaled)
            
        return model.classes_[np.argmax(predictions, axis=1)]
    
    def predict_proba(self, X):
        """Retorna probabilidades promedio ponderadas."""
        X_scaled = self.scaler.transform(X)
        predictions = np.zeros((X_scaled.shape[0], len(np.unique(next(iter(self.trained_models.values())).classes_))))
        
        for name, model in self.trained_models.items():
            predictions += self.weights[name] * model.predict_proba(X_scaled)
            
        return predictions / sum(self.weights.values())
    
    def save_model(self, path):
        """Guarda el ensemble entrenado."""
        model_data = {
            'scaler': self.scaler,
            'trained_models': self.trained_models,
            'weights': self.weights
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load_model(cls, path):
        """Carga un ensemble guardado."""
        classifier = cls()
        model_data = joblib.load(path)
        classifier.scaler = model_data['scaler']
        classifier.trained_models = model_data['trained_models']
        classifier.weights = model_data['weights']
        return classifier