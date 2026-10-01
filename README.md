# TP — Regresión Softmax en 20 Newsgroups

**Estudiante:** ÁlvaroGCh · **Legajo:** 61621

El trabajo completo está en [RESUELTO_Softmax.ipynb](RESUELTO_Softmax.ipynb): implementación de los tres TODO, explicaciones, resultados de la búsqueda de hiperparámetros, análisis de dinámica y tiempos, evaluación final y respuestas a las diez preguntas.

Se sigue el [notebook inicial de la consigna](https://github.com/cselmo/TP-Regresion-Softmax-TNG/blob/9267a2ff7839aad6fdceda2185baedabca9a4697/01_Softmax_Regression_20Newsgroups.ipynb). Se mantienen NLTK, TF-IDF, una capa lineal, CrossEntropyLoss, Adam, los rangos de hiperparámetros y los presupuestos originales: quince ensayos de cinco épocas y veinte épocas finales. Las comparaciones adicionales, las semillas y las mediciones se explican junto al código.

## Archivos

| Archivo | Contenido |
|---|---|
| `RESUELTO_Softmax.ipynb` | Código, salidas, gráficos, explicaciones y respuestas. |
| `README.md` | Instrucciones de lectura y ejecución. |
| `requirements.txt` | Dependencias directas. |
| `requirements-lock.txt` | Versiones del entorno utilizado en Windows con Python 3.12. |
| `.gitignore` | Exclusión del entorno y de los datos y registros generados. |

## Leer los resultados

Abrir el notebook en JupyterLab o en un visor de notebooks permite leer el trabajo con sus salidas y figuras guardadas. No hace falta ejecutar código para leerlo.

## Preparar el entorno y ejecutar

Se utilizó Python 3.12 de 64 bits y PyTorch para CPU. En Windows, abrir PowerShell **en la carpeta que contiene estos cinco archivos** y ejecutar:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch==2.14.0+cpu" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m jupyterlab RESUELTO_Softmax.ipynb
```

Seleccionar el kernel **Python 3 (ipykernel)** del entorno recién creado. Para reproducir todo, usar **Kernel → Restart Kernel and Run All Cells**. Para inspeccionar por partes, ejecutar las celdas de arriba hacia abajo con Shift + Enter.

La instalación y la primera descarga de recursos requieren conexión a Internet. El propio notebook descarga 20 Newsgroups y los recursos NLTK en `datos/`. No depende de archivos de resultados preexistentes ni de otro notebook. Cada ejecución completa vuelve a calcular el preprocesamiento y entrena los 21 modelos: 15 ensayos de búsqueda, 5 comparaciones y 1 final, con 195 épocas en total. La reutilización de las seis variantes de texto ocurre en memoria dentro de cada ejecución.

El lock documenta el entorno de Windows usado para las salidas incluidas; no es un lock universal para todos los sistemas. Para otro sistema, instalar la variante de [PyTorch](https://pytorch.org/get-started/locally/) apropiada y las dependencias de `requirements.txt`; diferencias de versiones o hardware pueden cambiar resultados y tiempos.

## Seguimiento de entrenamientos

MLflow y TensorBoard forman parte de la consigna. El notebook genera automáticamente `salidas/`, con la base de MLflow, el modelo final exportado y los eventos de TensorBoard. Cada entrenamiento tiene una corrida independiente. Esas carpetas se producen al ejecutar y no se necesitan para leer las salidas guardadas del notebook.

Para ver TensorBoard, desde otra terminal en la misma carpeta:

```powershell
.\.venv\Scripts\tensorboard.exe --logdir salidas/runs
```

Para consultar MLflow:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///salidas/mlflow.db
```

Abrir la dirección local que imprima cada herramienta. Ningún servicio necesita estar abierto para que el notebook entrene y genere sus figuras.

## Criterios de análisis

- Train y validación se separan antes de ajustar TF-IDF; test no interviene en la búsqueda.
- Optuna maximiza la accuracy de validación de la quinta época; no el máximo histórico ni F1.
- Las cinco comparaciones usan la misma representación y partición, cambiando LR o batch de a un factor. Se completan veinte épocas sin early stopping.
- Estabilización operativa: tres épocas sin una mejora de val_loss de al menos 0.001 respecto de la referencia. Se mide también tiempo hasta accuracy ≥ 0.70. Estos criterios no prueban convergencia matemática.
- Los tiempos de descarga, preprocesamiento, vectorización y entrenamiento no se confunden; el notebook especifica qué incluye cada intervalo.
- En el entrenamiento final la función evalúa test por época, como en el template. Esas curvas se describen sin usarlas para modificar hiperparámetros o elegir retrospectivamente la época final.

Las salidas se obtuvieron en CPU Intel Core i5-7200U, con dos hilos de PyTorch y semilla 42. Los análisis numéricos corresponden a la ejecución incluida. Al repetirla, los tiempos pueden variar con el equipo y su carga; una semilla no garantiza igualdad entre versiones y dispositivos. Se realizaron una búsqueda y una corrida por caso, sin estimar variabilidad entre semillas.
