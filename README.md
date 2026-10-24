# Detección de fraude con aprendizaje supervisado

Proyecto de INFO1185 — Inteligencia Artificial. Compara k-NN, SVM, Árbol de
Decisión y un Random Forest propio para detectar transacciones fraudulentas. El
criterio principal es minimizar falsos negativos: un fraude no detectado tiene
mayor prioridad que la exactitud global.

## Datos y alcance experimental

Se utiliza [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):
284.807 transacciones, 30 predictores (`Time`, `Amount`, `V1`–`V28`) y la
variable objetivo `Class` (`1` = fraude). El CSV no se versiona; debe ubicarse en:

```text
data/raw/creditcard.csv
```

Los resultados incluidos en `results/` corresponden a una muestra estratificada
reproducible de 30.000 registros: 29.948 legítimos y 52 fraudes. La división
reservó 70 % para entrenamiento (21.000) y 30 % para prueba (9.000: 8.984
legítimos y 16 fraudes).

## Estructura

```text
data/raw/                 dataset original esperado
notebooks/                exploración inicial
src/data/                 carga, limpieza y división estratificada
src/features/             selección por información mutua
src/imbalance/            análisis y sobremuestreo aleatorio
src/models/               cuatro clasificadores y pipelines
src/validation/           folds y búsqueda de hiperparámetros
src/evaluation/           métricas, comparación y figuras finales
scripts/                  ejecución de validación y evaluación
results/                  tablas y figuras del experimento de 30.000 filas
presentation/             presentación Beamer y guía de defensa
tests/                    pruebas automatizadas
```

## Instalación

Requiere Python 3.10 o superior. Las versiones admitidas están fijadas por rango
en `requirements.txt`: NumPy 1.26–2.x, pandas 2.x–3.x, scikit-learn 1.5–1.x,
Matplotlib 3.9–3.x, seaborn 0.13.x, Jupyter 1.x y pytest 8.x–9.x.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducción

Todos los pasos usan `RANDOM_STATE = 42`. El orden recomendado desde la raíz es:

```bash
# 1. Exploración
jupyter notebook notebooks/01_exploracion.ipynb

# 2. Búsqueda de hiperparámetros; el test permanece reservado
python scripts/run_issue_7.py --sample-size 30000

# 3. Reentrenamiento y evaluación final sobre test
python scripts/run_issue_8.py --sample-size 30000

# 4. Pruebas
pytest -q
```

Para usar las 284.807 filas, sustituya `--sample-size 30000` por
`--full-dataset` en los pasos 2 y 3. Esto genera un experimento distinto al que
respalda la presentación incluida.

La muestra estratificada, la división 70/30, los cinco folds y todos los
estimadores aleatorios usan la misma semilla. Los predictores duplicados se
mantienen juntos para impedir que una copia exacta cruce entrenamiento,
validación o prueba. No se garantiza reproducibilidad bit a bit entre versiones
distintas de las dependencias; conserve el entorno si necesita esa equivalencia.

## Modelos y metodología

- **k-NN:** clasifica por vecinos cercanos; requiere escalado.
- **SVM:** aprende una frontera de separación; se evaluaron kernels lineal y RBF.
- **Árbol de Decisión:** aplica particiones jerárquicas; no requiere escalado.
- **Random Forest propio:** entrena árboles independientes sobre muestras
  bootstrap, usa `max_features` en cada división y decide por mayoría.

Se seleccionan 10 variables mediante información mutua. El escalado, el selector
y el `RandomOverSampler` se ajustan exclusivamente con el training fold. Las ocho
condiciones (cuatro modelos, con y sin compensación) comparten validación cruzada
estratificada de cinco particiones. La búsqueda maximiza sensibilidad media y
usa precisión media para desempatar. Solo después se reentrena cada configuración
con todo el 70 % y se evalúa una vez en el 30 % reservado.

La selección por validación fue **SVM lineal con sobremuestreo** (`C=0,5`):
sensibilidad `0,9179 ± 0,0674`. En test obtuvo sensibilidad `0,8750` (14/16
fraudes), FNR `0,1250`, especificidad `0,9738` y precisión `0,0562`, con 235
falsas alarmas. Las tablas completas y la procedencia de cada métrica están en
`results/tables/`.

## Presentación

La fuente Beamer está en [`presentation/main.tex`](presentation/main.tex) y la
copia compilada en [`presentation/main.pdf`](presentation/main.pdf). La guía de
compilación, distribución del tiempo y preguntas de defensa está en
[`presentation/README.md`](presentation/README.md).
