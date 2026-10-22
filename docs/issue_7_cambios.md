# Corrección de Issue #7: desbalance, validación y rendimiento

## Qué ocurría

La búsqueda evaluaba ocho condiciones experimentales: k-NN, SVM, Árbol de
Decisión y Random Forest propio, cada uno sin y con sobremuestreo. En cada
candidato y fold se volvía a ajustar el escalador y el selector de las diez
variables por información mutua, aunque varias configuraciones usaban
exactamente el mismo conjunto de entrenamiento y preprocesamiento.

Con cuatro candidatos por condición y cinco folds, aquello implicaba 160
ajustes durante la validación cruzada, más ocho ajustes finales: 168 en total.
Además, el espacio de búsqueda del Random Forest propio incluía 10 árboles, una
cantidad par que puede provocar empates en la votación.

## Qué se cambió

- Se sustituyó el valor 10 por 11 en los candidatos de `CustomRandomForest`.
  La búsqueda valida que todo `n_estimators` sea impar; la regla de desempate
  propia del modelo sigue disponible como protección adicional.
- Se añadió una caché local a cada ejecución de la búsqueda. Su clave incluye el
  tipo de escalado y el fold; por tanto, solo reutiliza transformaciones cuando
  las filas de entrenamiento son idénticas.
- k-NN y SVM comparten la transformación estándar de cada fold. Árbol y Random
  Forest propio comparten la transformación sin escalado. El selector se ajusta
  exclusivamente con el training fold y transforma validación sin ajustarse de
  nuevo.
- El sobremuestreo ocurre después de transformar el entrenamiento y nunca toca
  validación ni test. También se reutiliza solo para el mismo fold transformado.
- El script incorpora `--sample-size N` (30.000 por defecto) y
  `--full-dataset`. La muestra es estratificada, reproducible (`random_state=42`)
  y se selecciona una vez para los cuatro modelos antes de la división 70/30.
- Se incorporaron CSV de tiempos y metadatos, además de las características
  seleccionadas y conteos de validación por fold.

La caché reduce los ajustes de preprocesamiento a 12: dos configuraciones de
preprocesamiento por cinco folds, más dos ajustes finales sobre el entrenamiento
completo. No se reutiliza ningún estimador entre folds ni se utiliza el conjunto
de prueba para seleccionar parámetros, variables o estrategias.

## Validación realizada

- Pruebas automatizadas: `42 passed`.
- Ejecución de humo con 10.000 registros: 16,61 s.
- Ejecución completa con 30.000 registros y las ocho condiciones: 140,32 s.
- Se evaluaron 32 candidatos en total y no se registraron advertencias de
  convergencia.

El cuello de botella restante es SVM balanceado: sus candidatos se entrenan con
los folds de entrenamiento sobremuestreados. La optimización elimina cálculos de
preprocesamiento repetidos, pero no omite clasificadores, kernels ni escenarios
necesarios para que la comparación sea válida.

## Artefactos generados

La ejecución escribe resultados locales en `results/tables/`:

- `issue_7_cv_results.csv`: una fila por modelo y condición de balanceo.
- `issue_7_cv_candidates.csv`: los 32 candidatos evaluados.
- `issue_7_fold_balance.csv`: conteos antes/después del muestreo y validación.
- `issue_7_timings.csv`: tiempos de preprocesamiento, sobremuestreo y condición.
- `issue_7_run_metadata.csv`: muestra, semilla, tamaños y tiempo total.

Estos resultados corresponden a la muestra de 30.000 registros. Para repetir el
experimento con todas las transacciones se debe ejecutar:

```bash
.venv/bin/python scripts/run_issue_7.py --full-dataset
```
