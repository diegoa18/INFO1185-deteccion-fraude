# Detección de fraude con aprendizaje supervisado

Proyecto de INFO1185 para explorar el conjunto **Credit Card Fraud Detection** y comparar, en etapas posteriores, clasificadores supervisados con prioridad en reducir falsos negativos de fraude.

El alcance actual incluye carga, exploración, diagnóstico, división reproducible, transformadores de preprocesamiento y selección de características. Los clasificadores, estrategias de balanceo, validación cruzada ejecutable, ajuste de hiperparámetros y evaluación comparativa todavía no están implementados.

## Estructura

```text
data/
  raw/                 # CSV original local, no versionado
  processed/           # salidas generadas, no versionadas
docs/                  # enunciado oficial
notebooks/
  01_exploracion.ipynb
src/
  config.py            # semillas, partición y rutas
  data/                # carga, diagnóstico y preparación
  features/            # selector de variables
tests/
  test_preprocessing.py
results/               # tablas y figuras futuras
```

Las carpetas restantes reservan espacio para las próximas issues y no contienen implementaciones anticipadas.

## Instalación

Se recomienda Python 3.10 o superior y un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Descargue **Credit Card Fraud Detection** desde Kaggle y ubique el archivo como:

```text
data/raw/creditcard.csv
```

El CSV no se descarga ni se versiona. Debe contener `Time`, `Amount`, `V1` a `V28` y `Class`; `Class` es el objetivo y nunca se incorpora como predictor.

## Exploración

Inicie Jupyter desde la raíz del proyecto y abra el notebook:

```bash
jupyter notebook notebooks/01_exploracion.ipynb
```

El notebook informa dimensiones, tipos, faltantes, duplicados, estadísticas, distribución de clases y gráficos de `Amount` y `Time`. Es estrictamente descriptivo: no borra filas ni aprende parámetros de escalado o selección. Los duplicados se diagnostican antes de decidir una limpieza, porque eliminarlos sin análisis podría afectar la clase minoritaria. La limpieza disponible en `clean_dataset` tampoco elimina duplicados salvo que se solicite explícitamente.

## Preparación sin fuga de información

La configuración central está en `src/config.py`:

- `RANDOM_STATE = 42`
- `TEST_SIZE = 0.30`
- `N_SPLITS = 5` (reservado para la validación cruzada de una issue posterior)

La división oficial es 70 % entrenamiento y 30 % prueba, con `stratify=y`. Aunque una descripción anterior de la issue mencionaba 70/15/15, se aplica el enunciado oficial: no se crea un conjunto de validación fijo. El conjunto de prueba se reserva intacto para la evaluación final.

```python
from src.data.load_data import load_dataset, separate_features_target
from src.data.preprocess import stratified_train_test_split

data = load_dataset()
X, y = separate_features_target(data)
X_train, X_test, y_train, y_test = stratified_train_test_split(X, y)
```

Use `create_scaler("standard")`, `create_scaler("minmax")` o `create_scaler("none")` según el clasificador. Las dos primeras opciones devuelven transformadores nuevos de scikit-learn; deben estar dentro de un pipeline para ajustarse solo con entrenamiento. `"none"` devuelve `"passthrough"`, apropiado para modelos basados en árboles.

## Selección de características

`make_mutual_information_selector(k=10)` crea un `SelectKBest` basado en información mutua, con semilla fija. Puede detectar asociaciones no lineales, pero no prueba causalidad ni sustituye la validación del modelo. Debe estar dentro del mismo pipeline que el escalado y ajustarse en cada partición de entrenamiento de validación cruzada; nunca se ajusta globalmente antes de validar.

```python
from sklearn.pipeline import Pipeline

from src.data.preprocess import create_scaler
from src.features.feature_selection import make_mutual_information_selector, selected_feature_names

pipeline = Pipeline([
    ("scale", create_scaler("standard")),
    ("select", make_mutual_information_selector(k=10)),
])
X_train_selected = pipeline.fit_transform(X_train, y_train)
X_test_selected = pipeline.transform(X_test)
names = selected_feature_names(pipeline.named_steps["select"], X_train.columns)
```

## Pruebas

Ejecute las pruebas unitarias sin requerir el CSV de Kaggle:

```bash
pytest
```

Cubren carga y validación del CSV, separación de `X` e `y`, división estratificada reproducible, prevención de `Class` como predictor, ajuste del escalador solo con entrenamiento e integración de escalado y selección en un pipeline.
