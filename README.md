# Detección de fraude con aprendizaje supervisado (INFO1185)

Proyecto para explorar el conjunto **Credit Card Fraud Detection** y comparar clasificadores supervisados (`k-NN`, `SVM`, `Árbol de Decisión` y `Random Forest` propio) con y sin compensación del desbalance. El objetivo prioritario es **minimizar la tasa de falsos negativos (FNR)** en la detección de fraude (`Class == 1`).

## 1. Dependencias y ejecución

Se requiere **Python 3.10+**. Las dependencias principales (listadas en `requirements.txt`) son `numpy>=1.26`, `pandas>=2.2`, `scikit-learn>=1.5`, `matplotlib>=3.9`, `seaborn>=0.13`, `jupyter>=1.1` y `pytest>=8.0`.

**Ubicación del dataset:** Descargue el archivo de Kaggle y ubíquelo en `data/raw/creditcard.csv`.

**Orden de ejecución (Semilla unificada RANDOM_STATE = 42):**
1. Exploración: `jupyter notebook notebooks/01_exploracion.ipynb`
2. Validación Cruzada (Issue 7): `python3 scripts/run_issue_7.py --sample-size 30000`
3. Evaluación Final (Issue 8): `python3 scripts/run_issue_8.py --sample-size 30000`

## 2. Metodología y Modelos Matemáticos

- **Métrica Prioritaria (FNR):** La Tasa de Falsos Negativos se define matemáticamente como la proporción de fraudes reales no detectados: `FNR = FN / (TP + FN) = 1 - Sensibilidad`.
- **Prevención de fuga de datos:** El escalado y la selección de características (`SelectKBest` con información mutua, `k=10`) se ajustan exclusivamente con el subconjunto de entrenamiento de cada fold de validación cruzada.
- **Tratamiento del desbalance:** Se contrastan configuraciones sin compensación frente al uso de `RandomOverSampler`, el cual duplica aleatoriamente la clase minoritaria solo en el pliegue de entrenamiento.
- **Random Forest Propio:** Implementado desde cero mediante bagging. Genera muestras bootstrap con reemplazo, aplica selección aleatoria de atributos por división (`max_features`) y combina predicciones por votación mayoritaria usando cantidades impares de árboles (`n_estimators = 5` o `11`) para evitar empates.

## 3. Resultados Experimentales (Muestra de 30.000 registros)

### Validación Cruzada (5 folds sobre el 70 % de entrenamiento)
Selección basada en maximizar la Sensibilidad media, usando la Precisión como desempate.

| Modelo | Condición | Sensibilidad (μ ± σ) | Especificidad (μ ± σ) | Precisión (μ ± σ) | F1-score (μ ± σ) | Exactitud (μ ± σ) | FNR (μ ± σ) |
|---|---|---|---|---|---|---|---|
| k-NN | Sin balanceo | 0,8321 ± 0,0593 | 0,9997 ± 0,0002 | 0,8281 ± 0,1029 | 0,8254 ± 0,0576 | 0,9994 ± 0,0002 | 0,1679 ± 0,0593 |
| k-NN | Con balanceo | 0,8321 ± 0,0593 | 0,9995 ± 0,0002 | 0,7590 ± 0,0816 | 0,7913 ± 0,0574 | 0,9992 ± 0,0002 | 0,1679 ± 0,0593 |
| SVM | Sin balanceo | 0,7464 ± 0,1692 | 0,9997 ± 0,0003 | 0,8614 ± 0,1240 | 0,7737 ± 0,0910 | 0,9993 ± 0,0002 | 0,2536 ± 0,1692 |
| **SVM** | **Con balanceo (Seleccionado)** | **0,9179 ± 0,0674** | 0,9713 ± 0,0116 | 0,0596 ± 0,0214 | 0,1109 ± 0,0368 | 0,9712 ± 0,0115 | **0,0821 ± 0,0674** |
| Árbol | Sin balanceo | 0,7250 ± 0,0746 | 0,9996 ± 0,0003 | 0,7679 ± 0,1354 | 0,7384 ± 0,0686 | 0,9991 ± 0,0003 | 0,2750 ± 0,0746 |
| Árbol | Con balanceo | 0,7786 ± 0,0655 | 0,9995 ± 0,0003 | 0,7367 ± 0,0921 | 0,7510 ± 0,0451 | 0,9991 ± 0,0003 | 0,2214 ± 0,0655 |
| CustomRF | Sin balanceo | 0,6357 ± 0,1974 | 0,9998 ± 0,0003 | 0,9048 ± 0,1313 | 0,7136 ± 0,1467 | 0,9992 ± 0,0003 | 0,3643 ± 0,1974 |
| CustomRF | Con balanceo | 0,7750 ± 0,1172 | 0,9998 ± 0,0003 | 0,8829 ± 0,1115 | 0,8105 ± 0,0507 | 0,9994 ± 0,0002 | 0,2250 ± 0,1172 |

### Evaluación en el Conjunto de Prueba Reservado (30 %)
Las 8 configuraciones reentrenadas sobre el 70% completo y evaluadas sobre el test set (8.984 legítimas, 16 fraudes).

| Modelo | Condición | Sensibilidad | Especificidad | Precisión | F1-score | Exactitud | FNR | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|
| k-NN | Sin balanceo | 0,8750 | 1,0000 | 1,0000 | 0,9333 | 0,9998 | 0,1250 | 8984 | 0 | 2 | 14 |
| k-NN | Con balanceo | 0,8750 | 0,9997 | 0,8235 | 0,8485 | 0,9994 | 0,1250 | 8981 | 3 | 2 | 14 |
| SVM | Sin balanceo | 0,6875 | 1,0000 | 1,0000 | 0,8148 | 0,9994 | 0,3125 | 8984 | 0 | 5 | 11 |
| **SVM** | **Con balanceo (Seleccionado)** | **0,8750** | 0,9738 | 0,0562 | 0,1057 | 0,9737 | **0,1250** | 8749 | 235 | 2 | 14 |
| Árbol | Sin balanceo | 0,5000 | 1,0000 | 1,0000 | 0,6667 | 0,9991 | 0,5000 | 8984 | 0 | 8 | 8 |
| Árbol | Con balanceo | 0,7500 | 0,9993 | 0,6667 | 0,7059 | 0,9989 | 0,2500 | 8978 | 6 | 4 | 12 |
| CustomRF| Sin balanceo | 0,6875 | 1,0000 | 1,0000 | 0,8148 | 0,9994 | 0,3125 | 8984 | 0 | 5 | 11 |
| CustomRF| Con balanceo | 0,7500 | 1,0000 | 1,0000 | 0,8571 | 0,9996 | 0,2500 | 8984 | 0 | 4 | 12 |

## 4. Análisis de Clasificadores Extremos

| Clasificador Base | TN | FP | FN | TP | Sensibilidad | Especificidad | Precisión | F1-score | Exactitud | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Siempre legítima (`Class = 0`) | 8984 | 0 | 16 | 0 | 0,0000 | 1,0000 | 0,0000 | 0,0000 | 0,9982 | 1,0000 |
| Siempre fraude (`Class = 1`) | 0 | 8984 | 0 | 16 | 1,0000 | 0,0000 | 0,0018 | 0,0035 | 0,0018 | 0,0000 |

**Conclusión Principal:**
El modelo SVM con balanceo detecta la mayor cantidad de fraudes (baja tasa FNR de 12,5%), pero a costa de un volumen severo de falsas alarmas (Precisión de 5,62% y 235 transacciones legítimas bloqueadas erróneamente). Además, el clasificador "siempre legítima" demuestra por qué la métrica de exactitud (99,82%) es inútil frente a desbalances extremos.
