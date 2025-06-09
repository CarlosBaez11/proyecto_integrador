import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from umap import UMAP
from hdbscan import HDBSCAN
import kaleido
import optuna
import os
import numpy as np
import plotly.io as pio
import pickle
from collections import defaultdict

class TopicModeler:
    def __init__(self, csv_path: str = None, text_column: str = 'text', model_path: str = 'bertopic_model', pickle_path: str = 'bertopic_model.pkl'):
        """
        csv_path: ruta al CSV para entrenamiento; si None, solo carga modelo.
        model_path: ruta para guardar/cargar modelo BERTopic nativo.
        pickle_path: ruta para guardar/cargar pickle del modelo.
        """
        self.csv_path = csv_path
        self.text_column = text_column
        self.model_path = model_path
        self.pickle_path = pickle_path
        self.df = None
        self.model = None
        self.topics = None
        self.probs = None
        self.embeddings = None
        self.sub_centroids = {}

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        if self.text_column not in self.df.columns:
            raise ValueError(f"La columna '{self.text_column}' no está en el archivo CSV.")
        self.df[self.text_column] = self.df[self.text_column].astype(str)
        print(f"Datos cargados. Total de registros: {len(self.df)}")

    def preprocess_text(self):
        self.df[self.text_column] = self.df[self.text_column].str.lower().str.strip()

    def _compute_embeddings(self, texts=None):
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        data = texts if texts is not None else self.df[self.text_column].tolist()
        return model.encode(data, show_progress_bar=True)

    # def tune_hdbscan(self, n_trials: int = 20):
    #     if self.embeddings is None:
    #         self.embeddings = self._compute_embeddings()

    #     def objective(trial):
    #         min_cluster_size = trial.suggest_int('min_cluster_size', 5, 100)
    #         min_samples = trial.suggest_int('min_samples', 1, 50)
    #         metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan'])
    #         clusterer = HDBSCAN(
    #             min_cluster_size=min_cluster_size,
    #             min_samples=min_samples,
    #             metric=metric,
    #             cluster_selection_method='eom',
    #             prediction_data=True
    #         )
    #         labels = clusterer.fit_predict(self.embeddings)
    #         mask = labels != -1
    #         if len(set(labels[mask])) < 2:
    #             return -1.0
    #         return silhouette_score(self.embeddings[mask], labels[mask])

    #     study = optuna.create_study(direction='maximize')
    #     study.optimize(objective, n_trials=n_trials)
    #     best = study.best_params
    #     return HDBSCAN(
    #         min_cluster_size=best['min_cluster_size'],
    #         min_samples=best['min_samples'],
    #         metric=best['metric'],
    #         cluster_selection_method='eom',
    #         prediction_data=True
    #     )

    def fit_model(self, tune_hyperparams: bool = True, n_trials: int = 20):
        """
        Entrena el modelo y calcula centroides de subtemas sin guardar.
        """
        if self.csv_path is None:
            raise ValueError("csv_path es requerido para entrenar modelo.")
        self.load_data()
        self.preprocess_text()
        self.embeddings = self._compute_embeddings()

        try :

            hdbscan_model = self.tune_hdbscan(n_trials) if tune_hyperparams else HDBSCAN(min_cluster_size=82, min_samples=26, metric='euclidean', prediction_data=True)

        except Exception:

            hdbscan_model = HDBSCAN(min_cluster_size=82, min_samples=26, metric='euclidean', prediction_data=True)

        self.model = BERTopic(
            embedding_model=None,
            vectorizer_model=CountVectorizer(ngram_range=(1, 2), stop_words="english"),
            umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine'),
            hdbscan_model=hdbscan_model,
            language="multilingual",
            calculate_probabilities=True
        )
        self.topics, self.probs = self.model.fit_transform(self.df[self.text_column], embeddings=self.embeddings)
        # Calcular centroides de subtemas
        hierarchy = self.model.hierarchical_topics(self.df[self.text_column])
        for parent, subs in hierarchy.items():
            for sub in subs:
                idxs = [i for i, t in enumerate(self.topics) if t == sub]
                if idxs:
                    self.sub_centroids[sub] = np.mean(self.embeddings[idxs], axis=0)
    def save_model(self):
            """
            Guarda el modelo en formatos .bertopic y .pkl una vez entrenado.
            """
            if not self.model:
                raise ValueError("No hay modelo para guardar.")
            self.model.save(f"{self.model_path}.bertopic")
            with open(self.pickle_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Modelos guardados en '{self.model_path}.bertopic' y '{self.pickle_path}'")

    def load_model(self, use_pickle: bool = False):
        """
        Carga el modelo desde .bertopic o .pkl según use_pickle.
        """
        if use_pickle:
            if not os.path.exists(self.pickle_path):
                raise ValueError("No se encontró el pickle del modelo.")
            with open(self.pickle_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            if not os.path.exists(f"{self.model_path}.bertopic"):
                raise ValueError("No se encontró el modelo BERTopic.")
            self.model = BERTopic.load(f"{self.model_path}.bertopic")
        print("Modelo cargado.")


    def get_topic_label(self, topic_id, n_words=2):
        """Etiqueta corta de tópico."""
        words = [w for w,_ in self.model.get_topic(topic_id)][:n_words]
        return ' '.join([w.capitalize() for w in words])

    def show_topics(self, n=10, n_label_words=2):
        """Muestra tópicos con etiqueta y palabras clave."""
        for tid, topic in self.model.get_topics().items():
            if tid==-1: continue
            label=self.get_topic_label(tid,n_label_words)
            words=[w for w,_ in topic][:n]
            print(f"Tema {tid} ({label}): {words}")

    def visualize_topics(self):
        """Gráfico interactivo de tópicos."""
        self.model.visualize_topics().show()

    def get_topic_hierarchy(self):
        """Retorna dict tópico->[subtemas]."""
        h = self.model.hierarchical_topics(self.df[self.text_column])
        # Desempaquetar tupla si existe
        if isinstance(h, tuple):
            h = h[0]
        # Convertir DataFrame a dict si fuera DataFrame
        if isinstance(h, pd.DataFrame):
            hierarchy = {}
            # Se asume columnas 'Parent' y 'Child'
            for _, row in h.iterrows():
                p = row.get('Parent') or row.get('parent')
                c = row.get('Child') or row.get('child')
                hierarchy.setdefault(p, []).append(c)
        elif isinstance(h, dict):
            hierarchy = h
        else:
            raise ValueError("Formato inesperado de la jerarquía de tópicos.")
        # Eliminar outliers
        hierarchy.pop(-1, None)
        return hierarchy

    def create_subtopics(self):
        """Muestra jerarquía de subtemas en gráfico interactivo."""
        # Usar la visualización interna sin pasar argumentos
        fig = self.model.visualize_hierarchy()
        fig.show()
        # Si se necesita la estructura, usar get_topic_hierarchy()
        return self.get_topic_hierarchy()

    def predict(self, texts, n_label_words=2):
        """Predice tópico, subtema y etiqueta."""

        macro_topics = {
            # Security & Medical Data Compliance
            0: "Security & Compliance",
            1: "Security & Compliance",
            
            # Technical Support & IT Issues
            -1: "Technical Support",
            12: "Technical Support",
            14: "Technical Support",
            15: "Technical Support",
            17: "Technical Support",
            19: "Technical Support",
            22: "Technical Support",
            27: "Technical Support",
            30: "Technical Support",
            32: "Technical Support",
            
            # Business Analytics & Investment
            3: "Business Analytics",
            9: "Business Analytics",
            20: "Business Analytics",
            29: "Business Analytics",
            33: "Business Analytics",
            
            # Digital Marketing & Branding
            2: "Digital Marketing",
            5: "Digital Marketing",
            6: "Digital Marketing",
            7: "Digital Marketing",
            31: "Digital Marketing",
            
            # Project Management & SaaS
            4: "Project Management",
            8: "Project Management",
            10: "Project Management",
            16: "Project Management",
            21: "Project Management",
            
            # Financial Operations
            7: "Financial Operations",
            13: "Financial Operations",
            28: "Financial Operations",
            
            # Database & System Integration
            23: "System Integration",
            24: "System Integration",
            25: "System Integration",
            
            # Infrastructure & Performance
            11: "Infrastructure",
            18: "Infrastructure",
            26: "Infrastructure",
            
            # German Language Support
            18: "German Support",
            19: "German Support",
            
            # Special Cases
            26: "Assistance Requests",
            30: "Browser Issues"
        }


        if not self.model:
            self.load_model(use_pickle=True)
        embs=self._compute_embeddings(texts)
        topics, probs=self.model.transform(texts,embeddings=embs)
        subts=[]
        labels=[]
        for emb,topic in zip(embs,topics):
            labels.append(self.get_topic_label(topic,n_label_words))
            # buscar subtema cercano
            cand={s:c for s,c in self.sub_centroids.items() if self.model.find_topic(s)[1]==topic}
            subts.append(min(cand, key=lambda s: np.linalg.norm(emb-cand[s])) if cand else None)

        print(topic)
        macro = macro_topics.get(topic, "Undefined Macro-topic")

        return macro,labels

    def compute_npmi_coherence(self, top_n_words: int = 10) -> float:
        """
        Calcula la coherencia de tópicos usando NPMI.
        """
        if self.df is None or self.model is None:
            raise ValueError("Se necesita cargar datos y entrenar/cargar el modelo.")
        
        # Extraer textos tokenizados
        texts = [text.split() for text in self.df[self.text_column]]

        # Extraer las top-n palabras por tópico
        topic_words = []
        for topic_id in range(len(self.model.get_topics())):
            if topic_id == -1:  # omitir outliers
                continue
            words = [word for word, _ in self.model.get_topic(topic_id)[:top_n_words]]
            topic_words.append(words)

        # Crear diccionario y corpus
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        # Modelo de coherencia
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=texts,
            dictionary=dictionary,
            coherence='c_npmi'
        )

        coherence_score = coherence_model.get_coherence()
        print(f"Coherencia NPMI: {coherence_score:.4f}")
        return coherence_score


    
# Ejemplo de uso en producción:
# if __name__ == '__main__':
#     from topic_modeler_optuna import TopicModeler
#     tm = TopicModeler(csv_path=None, model_path='bertopic_model', pickle_path='bertopic_model.pkl')
#     tm.load_model(use_pickle=True)
#     nuevos_textos = ["Tengo problemas para entrar a mi cuenta"]
#     topics, subtopics, labels, probabilities = tm.predict(nuevos_textos)
#     for text, topic, subtopic, label, prob in zip(nuevos_textos, topics, subtopics, labels, probabilities):
#         print(f"Texto: {text}")
#         print(f"  - Etiqueta: {label}")
#         print(f"  - Tópico: {topic}")
#         print(f"  - Subtema: {subtopic}")
#         print(f"  - Probabilidad: {prob}\n")