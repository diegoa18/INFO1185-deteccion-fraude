# Detección de fraude con aprendizaje supervisado

Proyecto de INFO1185 para explorar el conjunto **Credit Card Fraud Detection** y comparar, en etapas posteriores, clasificadores supervisados con prioridad en reducir falsos negativos de fraude.

El alcance actual incluye carga, exploración, diagnóstico, división reproducible, transformadores de preprocesamiento, selección de características, configuración reutilizable de validación cruzada y un pipeline k-NN. Las estrategias de balanceo, búsqueda de hiperparámetros y evaluación comparativa todavía no están implementadas.

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
  models/              # pipelines de clasificadores
  validation/          # particiones estratificadas de validación cruzada
tests/
  test_preprocessing.py
  test_knn.py
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

La división oficial es 70 % entrenamiento y 30 % prueba, estratificada por `Class`. Aunque una descripción anterior de la issue mencionaba 70/15/15, se aplica el enunciado oficial: no se crea un conjunto de validación fijo. El conjunto de prueba se reserva intacto para la evaluación final. Además, la división mantiene los registros con predictores idénticos juntos para que una copia exacta no aparezca en entrenamiento y prueba a la vez.

```python
from src.data.load_data import load_dataset, separate_features_target
from src.data.preprocess import stratified_train_test_split

data = load_dataset()
X, y = separate_features_target(data)
X_train, X_test, y_train, y_test = stratified_train_test_split(X, y)
```

Use `create_scaler("standard")`, `create_scaler("minmax")` o `create_scaler("none")` según el clasificador. Las dos primeras opciones devuelven transformadores nuevos de scikit-learn; deben estar dentro de un pipeline para ajustarse solo con entrenamiento. `"none"` devuelve `"passthrough"`, apropiado para modelos basados en árboles.

El dataset oficial no presenta faltantes según la exploración. Si se trabajara con una variante que sí los tuviera, la imputación debe ser otro transformador del pipeline, ajustado dentro de cada fold, nunca una modificación global antes de la división. Los tipos inesperados y valores inconsistentes se informan en el diagnóstico para decidir una regla de dominio explícita; no se coercionan silenciosamente. Los duplicados exactos se conservan por defecto y se diagnostican por separado: eliminarlos puede alterar la clase minoritaria. Si se decide eliminarlos, esa decisión debe tomarse antes de la partición y justificarse con el impacto por clase.

## Validación cruzada

La validación cruzada se configura únicamente sobre `X_train` e `y_train`; no crea un tercer conjunto fijo ni toca `X_test`/`y_test`. Para que modelos y condiciones se comparen con exactamente los mismos folds, materialice los índices una sola vez y reutilícelos:

```python
from src.validation import materialize_stratified_folds

folds = materialize_stratified_folds(X_train, y_train)
for train_index, validation_index in folds:
    X_fold_train = X_train.iloc[train_index]
    y_fold_train = y_train.iloc[train_index]
    X_fold_validation = X_train.iloc[validation_index]
    y_fold_validation = y_train.iloc[validation_index]
```

Cada pipeline debe ajustarse dentro de ese ciclo sobre `X_fold_train`/`y_fold_train`; así, escalado y selección quedan aislados de los conjuntos de validación y prueba.

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

## k-Nearest Neighbors

`create_knn_pipeline` mantiene el escalado y la selección dentro del pipeline,
por lo que sus parámetros se ajustan únicamente al invocar `fit` sobre los datos
de entrenamiento o sobre cada fold de entrenamiento. El escalado estándar se usa
por defecto porque k-NN calcula distancias.

```python
from src.models.knn import create_knn_pipeline

pipeline = create_knn_pipeline(n_neighbors=5, metric="minkowski")
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

Por defecto, ``k_features="all"`` conserva todas las características. Para
seleccionar una cantidad específica mediante información mutua, entregue un
entero positivo, por ejemplo ``k_features=10``.

Los hiperparámetros del clasificador quedan accesibles como
`classifier__n_neighbors`, `classifier__metric`, `classifier__weights` y
`classifier__p` mediante `get_params` y `set_params`.

## Pruebas

Ejecute las pruebas unitarias sin requerir el CSV de Kaggle:

```bash
pytest
```

Cubren carga y validación del CSV, separación de `X` e `y`, división estratificada reproducible 70/30, ausencia de solapamiento por índice, prevención de `Class` como predictor, ajuste de transformadores solo con entrenamiento e integración de escalado y selección dentro de pipelines y cinco folds estratificados.
