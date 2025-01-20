from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class SpeakerClassifierRF:
    def __init__(self, classifier_type='xgb', **kwargs):
        """
        Inicializa el clasificador.
        Args:
            classifier_type: 'rf' para Random Forest, 'xgb' para XGBoost
            **kwargs: Argumentos específicos para cada clasificador
        """
        self.classifier_type = classifier_type
        
        if classifier_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                **kwargs
            )
        elif classifier_type == 'xgb':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                **kwargs
            )
        else:
            raise ValueError("Clasificador no soportado. Use 'rf' o 'xgb'")
            
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def train(self, X, y):
        """Entrena el modelo seleccionado."""
        # Codificar las etiquetas
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Dividir los datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Escalar los datos
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar el modelo
        print(f"Entrenando modelo {self.classifier_type.upper()}...")
        self.model.fit(X_train_scaled, y_train)
        
        # Realizar predicciones
        y_pred = self.model.predict(X_test_scaled)
        
        # Obtener las etiquetas originales para el reporte
        y_test_original = self.label_encoder.inverse_transform(y_test)
        y_pred_original = self.label_encoder.inverse_transform(y_pred)
        
        # Imprimir métricas
        print("\nReporte de clasificación:")
        print(classification_report(y_test_original, y_pred_original))
        
        # Crear matriz de confusión
        cm = confusion_matrix(y_test_original, y_pred_original)
        
        # Visualizar matriz de confusión
        plt.figure(figsize=(10, 8))
        classes = sorted(set(self.label_encoder.classes_))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes,
                    yticklabels=classes)
        plt.title(f'Matriz de Confusión - {self.classifier_type.upper()}')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.show()
        
        # Si es Random Forest, mostrar importancia de características
        if self.classifier_type == 'rf':
            self._plot_feature_importance()
            
        return X_test_scaled, y_test_original, y_pred_original
    
    def _plot_feature_importance(self):
        """Visualiza la importancia de las características para Random Forest."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(importances)), importances)
            plt.title('Importancia de Características')
            plt.xlabel('Índice de Característica')
            plt.ylabel('Importancia')
            plt.show()
    
    def predict(self, X):
        """Realiza predicciones con el modelo entrenado."""
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred)
    
    def predict_proba(self, X):
        """Realiza predicciones de probabilidad con el modelo entrenado."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save_model(self, path):
        """Guarda el modelo entrenado."""
        model_data = {
            'scaler': self.scaler,
            'model': self.model,
            'classifier_type': self.classifier_type,
            'label_encoder': self.label_encoder
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load_model(cls, path):
        """Carga un modelo guardado."""
        model_data = joblib.load(path)
        model = cls(classifier_type=model_data['classifier_type'])
        model.scaler = model_data['scaler']
        model.model = model_data['model']
        model.label_encoder = model_data['label_encoder']
        return model