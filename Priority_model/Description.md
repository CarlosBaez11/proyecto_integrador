
# Priocheck_up.ipynb

## Descripción

Este notebook realiza un análisis y modelado para la **clasificación de la prioridad de tickets de soporte multilingües**. Incluye limpieza de datos, generación de representaciones vectoriales (embeddings), manejo de desbalanceo en clases y entrenamiento diferentes modelos de machine learning para hallar el que mejor desempeño logra para la tarea especifica con el conjunto de datos disponible

## Objetivo del modelo

Clasificar correctamente la prioridad de un ticket a partir de su contenido textual y metadatos relevantes, utilizando modelos supervisados.

## Estructura del Notebook

1. **Carga y Exploración de Datos**
   - Lectura de un CSV con pandas.
   - Revisión de tipos de datos y valores faltantes.

2. **Preprocesamiento**
   - Eliminación de registros nulos.
   - Unificación y limpieza de texto.
   - Análisis de clases desbalanceadas.

3. **Generación de Embeddings**
   - Uso de `sentence-transformers`, en este caso "paraphrase-multilingual-MiniLM-L12-v2" para convertir el texto en vectores numéricos y así facilitar su procesamiento 

4. **División del Conjunto de Datos**
   - Separación en conjuntos de entrenamiento y prueba.

5. **Balanceo de Clases**
   - Uso de técnicas como `SMOTE` para tratar el desbalanceo de clases que estaba marcado en el conjunto de datos 
Antes del SMOTE: [7198 3787 7552]
Después del SMOTE: [7552 7552 7552]

6. **Entrenamiento de Modelos**
   - Comparación entre múltiples clasificadores: XGBoost, LightGBM, CatBoost, entre otros.
   - Evaluación con métricas como precisión, recall y f1-score.

7. **Resultados**
   - Métricas por clase.
   - Matriz de confusión y reporte de clasificación.

7. **Descarga PKL**
   - Se descarga el archivo de configuración del modelo para ser usado en el despliegue de este, más información en: 
   proyecto_integrador/deployment/

## Requisitos

Instalar los siguientes paquetes:

```bash
pip install sentence-transformers xgboost lightgbm catboost scikit-learn imbalanced-learn
```

## Dataset

El archivo de datos se encuentra en:
```
data/dataset-tickets-multi-lang-4-20k.csv
```