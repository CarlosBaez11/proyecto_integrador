from collections import defaultdict
import math
import pandas as pd
from bertopic import BERTopic

def compute_npmi(df: pd.DataFrame, model: BERTopic, text_column: str = "text", top_n_words: int = 10, window_size: int = 10):
    """
    Calcula la NPMI (Normalized Pointwise Mutual Information) promedio por tema generado por un modelo BERTopic.
    
    Args:
        df (pd.DataFrame): DataFrame con la columna de texto original.
        model (BERTopic): Modelo ya entrenado de BERTopic.
        text_column (str): Nombre de la columna que contiene texto.
        top_n_words (int): Número de palabras principales por tema a evaluar.
        window_size (int): Tamaño de la ventana para conteo de coocurrencias.
    """
    print("\nCalculando Normalized Pointwise Mutual Information (NPMI)...")

    tokenized_docs = df[text_column].astype(str).str.lower().str.split()

    word_freq = defaultdict(int)
    cooccur_freq = defaultdict(int)
    total_windows = 0

    for tokens in tokenized_docs:
        for i in range(len(tokens)):
            window = tokens[i:i + window_size]
            if len(window) < 2:
                continue
            total_windows += 1
            for w in window:
                word_freq[w] += 1
            for i in range(len(window)):
                for j in range(i + 1, len(window)):
                    pair = tuple(sorted((window[i], window[j])))
                    cooccur_freq[pair] += 1

    topic_keywords = model.get_topics()
    topic_npmi_scores = {}

    for topic_id, words in topic_keywords.items():
        if topic_id == -1:
            continue  # Ignorar outliers
        selected_words = [word for word, _ in words[:top_n_words]]
        score = 0
        count = 0
        for i in range(len(selected_words)):
            for j in range(i + 1, len(selected_words)):
                w1, w2 = selected_words[i], selected_words[j]
                pair = tuple(sorted((w1, w2)))
                if pair in cooccur_freq:
                    p_xy = cooccur_freq[pair] / total_windows
                    p_x = word_freq[w1] / total_windows
                    p_y = word_freq[w2] / total_windows
                    if p_x > 0 and p_y > 0 and p_xy > 0:
                        pmi = math.log2(p_xy / (p_x * p_y))
                        npmi = pmi / -math.log2(p_xy)
                        score += npmi
                        count += 1
        avg_npmi = score / count if count > 0 else 0
        topic_npmi_scores[topic_id] = avg_npmi

    print("\nNPMI promedio por tema:")
    for topic_id, score in sorted(topic_npmi_scores.items(), key=lambda x: -x[1]):
        print(f"Tema {topic_id}: NPMI promedio = {score:.4f}")
