# Detección de fraude con aprendizaje supervisado

Proyecto de INFO1185 para explorar el conjunto **Credit Card Fraud Detection** y comparar, en etapas posteriores, clasificadores supervisados con prioridad en reducir falsos negativos de fraude.

El alcance actual incluye carga, exploración, diagnóstico, división reproducible, transformadores de preprocesamiento, selección de características, validación cruzada común, compensación del desbalance y búsqueda de hiperparámetros para k-NN, SVM, Árbol de Decisión y Random Forest propio. La evaluación final sobre test se mantiene como una etapa separada.

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
  imbalance/           # análisis y sobremuestreo sin fuga de información
  models/              # pipelines de clasificadores
  validation/          # folds y optimización comparativa
tests/
  test_preprocessing.py
  test_knn.py
  test_issue_7.py
results/               # tablas y figuras futuras
```

Las carpetas restantes reservan espacio para resultados generados localmente.

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
- `N_SPLITS = 5` (validación cruzada estratificada de Issue #7)

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

## Issue #7: desbalance, validación y optimización

En el CSV oficial local, la distribución es la siguiente (la clase positiva es
siempre `Class == 1`, fraude):

| Clase | Significado | Cantidad | Porcentaje |
|---:|---|---:|---:|
| 0 | legítima | 284 315 | 99.827251 % |
| 1 | fraude | 492 | 0.172749 % |

La razón aproximada es **577.88 transacciones legítimas por cada fraude**. Por
ello accuracy no sirve como métrica de selección: un clasificador que respondiera
siempre clase 0 alcanzaría cerca de 99.83 % de accuracy y, aun así, tendría recall
0 para fraude. La optimización prioriza el recall/sensibilidad de `Class == 1`;
si dos configuraciones empatan dentro de `1e-9`, se escoge la de mayor precision
de fraude.

La ejecución predeterminada usa una muestra estratificada reproducible de 30 000
registros, suficiente para conservar fraudes en el entrenamiento y sus cinco
folds. El muestreo ocurre antes de la única división 70/30; no ajusta
transformadores y el mismo subconjunto se entrega a los cuatro clasificadores:

```bash
.venv/bin/python scripts/run_issue_7.py --sample-size 30000
```

También puede usar el dataset completo, con el coste considerablemente mayor de
SVC con kernel RBF y del Random Forest propio:

```bash
.venv/bin/python scripts/run_issue_7.py --full-dataset
```

`--sample-size` admite valores desde 10 000 y emplea `random_state=42`. El
script registra el modo, la semilla, el tamaño, los conteos por clase del origen,
la muestra, entrenamiento y test, el total de candidatos y el tiempo total real.
Los resultados con muestra deben interpretarse
como resultados de esa muestra, no como resultados del dataset completo.

El script realiza únicamente la selección usando el 70 % de entrenamiento y deja
el 30 % de prueba aislado. Genera localmente, sin versionarlos, estos artefactos:

- `results/tables/issue_7_class_imbalance.csv`
- `results/tables/issue_7_source_class_imbalance.csv`
- `results/figures/issue_7_class_distribution.png`
- `results/tables/issue_7_cv_results.csv` (las ocho configuraciones finales)
- `results/tables/issue_7_cv_candidates.csv` (todos los candidatos evaluados)
- `results/tables/issue_7_fold_balance.csv` (conteos de entrenamiento y validación por fold)
- `results/tables/issue_7_run_metadata.csv` (muestra, clases y semilla)
- `results/tables/issue_7_timings.csv` (preprocesamiento, sobremuestreo y condiciones)

### Validación y prevención de fuga

Se crea una sola vez y se reutiliza para los cuatro modelos, con y solo con el
conjunto de entrenamiento:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

`materialize_stratified_folds` congela los mismos índices antes de iniciar las
ocho búsquedas. Cada llamada a `fit` sigue este orden dentro del training fold:

```text
escalado (cuando corresponde) → selección por información mutua
→ RandomOverSampler opcional → clasificador
```

La selección fija las 10 variables con mayor información mutua para esta ronda
de experimentos, pero aprende esas variables nuevamente dentro de cada training
fold; no se ajusta una vez sobre el dataset completo. Para evitar cálculos
redundantes, la búsqueda conserva una caché local por `(tipo de escalado, fold)`:
se ajustan y transforman una sola vez las filas de entrenamiento y validación, y
los candidatos que comparten esas mismas filas reutilizan el resultado. k-NN y
SVM comparten la clave con escalado estándar; Árbol y Random Forest propio la
comparten sin escalado. La caché nunca se comparte entre folds, ni con test. En
la búsqueda completa esto reduce los ajustes de selector/escalador de 168 (8
condiciones × 4 candidatos × 5 folds, más 8 ajustes finales) a 12 (dos tipos de
preprocesamiento × cinco folds, más dos ajustes finales).

El `RandomOverSampler` propio se utiliza porque `imbalanced-learn` no es una
dependencia del proyecto. Conserva todas las muestras de entrenamiento y duplica
aleatoriamente las minoritarias hasta igualar la clase mayoritaria, usando
`random_state=42`. Nunca se ajusta sobre validación o test. El mismo pipeline
también garantiza que el scaler y el selector se ajusten solamente con los datos
del training fold. Para árboles y el Random Forest propio el escalado es
`passthrough`; k-NN y SVM usan escalado estándar.

El resultado principal incluye explícitamente `class_0_before`, `class_1_before`,
`class_0_after` y `class_1_after`; corresponden al ajuste final sobre el 70 % de
entrenamiento. También registra características seleccionadas, cantidad de
candidatos, recall medio/desviación estándar, precision, y tiempos de búsqueda y
ajuste final. La tabla por fold contiene además los conteos de validación para
verificar que los validation folds nunca fueron balanceados.

### Espacios de búsqueda acotados

Las configuraciones viven en `DEFAULT_SEARCH_SPACES` de
`src/validation/experiments.py`. Son candidatas deliberadamente acotadas para
evitar una explosión de costo con 284 807 filas, y cada una se evalúa con los
mismos cinco folds.

| Modelo | Hiperparámetros explorados |
|---|---|
| k-NN | `n_neighbors` = 3, 5, 7, 11; `weights` = uniform/distance; `metric` = euclidean/manhattan en cuatro combinaciones representativas |
| SVM | `kernel` = linear/rbf; `C` = 0.5/2.0; `gamma` = scale/0.1 solo para RBF (sin combinaciones inválidas) |
| Árbol | `max_depth` = None/12; `min_samples_split` = 2/20; `min_samples_leaf` = 1/5; `criterion` = gini/entropy en cuatro combinaciones representativas |
| Random Forest propio | `n_estimators` = 5/11; `max_depth` = None/12; `max_features` = sqrt/log2 en cuatro combinaciones representativas |

El Random Forest se mantiene como `CustomRandomForest`; no se sustituye por la
implementación de scikit-learn ni por `BaggingClassifier`. Su semilla permanece
fija en 42 para que la búsqueda sea comparable y reproducible, no como una
variable que pueda alterar artificialmente la selección. Todos los candidatos
usan una cantidad impar de árboles (5 u 11), evitando empates en la votación
mayoritaria; la regla de desempate del modelo sigue presente como defensa
adicional. La búsqueda rechaza explícitamente cualquier candidato par.

SVC conserva los kernels `linear` y `rbf`, con `C=0.5/2.0`; `gamma` se explora
solo para RBF. No se establece un límite artificial de iteraciones: un ajuste no
convergido no se oculta como resultado válido. La búsqueda registra advertencias
de convergencia por candidato y añade contexto de modelo, condición, candidato y
fold a cualquier error. El costo dominante restante es RBF sobre el training set
sobremuestreado; la caché reduce información mutua y escalado, pero no elimina
entrenamientos necesarios para mantener la comparación experimental.

`issue_7_timings.csv` contiene el tiempo real de cada preprocesamiento y
sobremuestreo compartido, además de búsqueda, ajuste final y tiempo por condición.
La consola muestra el tiempo total de ejecución. Las ocho filas de
`issue_7_cv_results.csv` se identifican por modelo y `balanced`, y mantienen los
campos necesarios para reutilizar la selección posterior.

## Pruebas

Ejecute las pruebas unitarias sin requerir el CSV de Kaggle:

```bash
pytest
```

Cubren carga y validación del CSV, separación de `X` e `y`, división y muestreo
estratificado reproducibles, ausencia de solapamiento por índice, prevención de
`Class` como predictor, ajuste de transformadores solo con entrenamiento,
integración de escalado y selección dentro de pipelines, cinco folds
compartidos, caché de transformaciones equivalente, sobremuestreo aislado al
training fold, las ocho condiciones, árboles impares y el desempate
recall/precision.
