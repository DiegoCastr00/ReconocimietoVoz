from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

class SpeakerClassifierSVM:
    def __init__(self):
        self.model = SVC(
            C=10,
            kernel='rbf',
            gamma=0.001,
            probability=True,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        
    def train(self, X, y):
        """Entrena el modelo SVM."""
        # Dividir los datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Escalar los datos
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar el modelo
        print("Entrenando modelo SVM...")
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        
        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred))
        
        cm = confusion_matrix(y_test, y_pred)

        # Graficar la matriz de confusión con las clases
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.model.classes_, 
                    yticklabels=self.model.classes_)
        plt.title('Matriz de Confusión')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.show()
        
        return X_test_scaled, y_test, y_pred
        
    def predict(self, X):
        """Realiza predicciones con el modelo entrenado."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Realiza predicciones de probabilidad con el modelo entrenado."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save_model(self, path):
        """Guarda el modelo entrenado."""
        model_data = {
            'scaler': self.scaler,
            'model': self.model
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load_model(cls, path):
        """Carga un modelo guardado."""
        model = cls()
        model_data = joblib.load(path)
        model.scaler = model_data['scaler']
        model.model = model_data['model']
        return model