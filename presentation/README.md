# Presentación final

Presentación Beamer de 11 diapositivas principales y 4 de respaldo, preparada
para una defensa de 10–12 minutos. Los números provienen de los CSV versionados
en `results/tables/`; `generate_figures.py` no entrena modelos ni altera esos
resultados.

## Compilación

En Debian/Ubuntu, el entorno completo se puede instalar con:

```bash
sudo apt install latexmk texlive-latex-recommended texlive-pictures \
  texlive-lang-spanish
```

Desde la raíz del repositorio:

```bash
# Regenerar las figuras desde los CSV de las issues 7 y 8
MPLCONFIGDIR=/tmp/info1185-matplotlib \
  .venv/bin/python presentation/generate_figures.py

# Compilar con latexmk (recomendado)
cd presentation
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Alternativa sin `latexmk`:

```bash
cd presentation
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

El resultado esperado es `presentation/main.pdf`. Se requieren Beamer, TikZ,
Booktabs, Babel en español y los paquetes LaTeX habituales. Para limpiar solo
archivos auxiliares: `latexmk -c`.

## Orden y tiempo sugeridos

| Tiempo | Expositor | Diapositivas | Enfoque |
|---:|---|---|---|
| 0:00–0:45 | Máximo | Portada | Problema y prioridad de FNR |
| 0:45–2:05 | Máximo | 1–2 | Datos y metodología sin fuga |
| 2:05–3:00 | Máximo | 3 | Diferencias entre clasificadores |
| 3:00–4:30 | Máximo | 4 | Random Forest propio |
| 4:30–5:20 | Máximo | 5 | Ocho condiciones y sobremuestreo |
| 5:20–6:20 | Diego | 6 | Búsqueda y criterio de selección |
| 6:20–7:30 | Diego | 7 | Resultados de validación cruzada |
| 7:30–8:45 | Diego | 8 | Resultados del test reservado |
| 8:45–10:10 | Diego | 9 | Análisis crítico |
| 10:10–11:15 | Diego | 10 | Conclusiones y limitaciones |

Las diapositivas de respaldo no se exponen salvo que surja una pregunta. Ambos
integrantes deben poder explicar cualquier sección; esta distribución organiza
el relato, no divide la responsabilidad técnica.

## Mensajes que no deben confundirse

- La configuración ganadora se eligió por validación cruzada, nunca por test.
- El test se evaluó una sola vez. Sus columnas `*_test_std` estiman dispersión
  mediante 200 remuestras bootstrap estratificadas de predicciones fijas; no
  representan cinco folds ni múltiples reentrenamientos.
- SVM con sobremuestreo fue seleccionada por su sensibilidad media de 0,9179 en
  CV. En test empató a k-NN en sensibilidad (0,8750), pero produjo 235 falsos
  positivos y una precisión de 0,0562.
- “Siempre fraude” minimiza FNR, pero genera 8.984 falsas alarmas; por eso la
  sensibilidad debe analizarse junto con precisión y especificidad.

## Preguntas probables de defensa

**¿Por qué no se seleccionó k-NN si en test obtuvo la misma sensibilidad y mejor
precisión?**  Porque el test estaba reservado. Elegir después de observarlo sería
fuga de información. SVM con balanceo tuvo la mayor sensibilidad media en CV.

**¿Dónde se evita la fuga de información?**  La división 70/30 ocurre antes del
ajuste. Dentro de cada fold, escalador y selector se ajustan solo con training;
el sobremuestreo también usa únicamente ese training. Validación y test solo se
transforman y predicen.

**¿Qué hace propio al Random Forest?**  El código genera para cada árbol una
muestra bootstrap de tamaño `n` con reemplazo, deriva una semilla distinta,
entrena un árbol con `max_features` aleatorio por división y combina clases por
voto mayoritario. No usa `RandomForestClassifier` ni `BaggingClassifier`.

**¿Por qué aparecen observaciones repetidas en un bootstrap?**  Cada una de las
`n` extracciones permite volver a elegir cualquier fila. Las repeticiones y las
omisiones producen diversidad entre árboles; el bagging promedia esa variación.

**¿En qué se diferencia un árbol del Random Forest propio?**  Un árbol aprende
una sola partición y tiene alta varianza. El bosque combina árboles entrenados
con filas y subconjuntos de atributos distintos. En este test, con balanceo,
ambos detectaron 12 fraudes, pero el árbol produjo 6 FP y el bosque 0.

**¿Qué hizo el tratamiento del desbalance?**  `RandomOverSampler` replicó al azar
los 36 fraudes del entrenamiento hasta igualar los 20.964 ejemplos legítimos.
Se aplicó de nuevo dentro de cada training fold. Mejoró la sensibilidad de SVM,
Árbol y RF en CV, pero en SVM redujo fuertemente la precisión.

**¿Por qué la exactitud es engañosa?**  Una regla “siempre legítima” alcanza
99,82 % porque casi todo es legítimo, aunque pierde los 16 fraudes del test:
sensibilidad 0 y FNR 1.

**¿Cómo se eligieron los hiperparámetros?**  Se evaluaron cuatro candidatos
acotados por modelo y condición sobre los mismos cinco folds. Se maximizó la
sensibilidad media y se usó precisión media para desempatar.

**¿Qué significan las desviaciones de test?**  Son la dispersión de las métricas
al remuestrear estratificadamente 200 veces las predicciones ya obtenidas. Miden
incertidumbre muestral condicionada al modelo ajustado; no miden variación de
entrenamiento.

**¿Cuál es la principal limitación?**  Solo hay 52 fraudes en la muestra y 16 en
test. Dos errores cambian la sensibilidad en 12,5 puntos porcentuales. Además,
faltan costos operativos, calibración, validación temporal y análisis de deriva.

**¿Qué mejorarían?**  Ejecutar el dataset completo, validar temporalmente,
calibrar probabilidades y fijar un umbral según el costo relativo de FN y FP.
También se puede comparar sobremuestreo, ponderación de clases y métodos
sintéticos dentro de CV.

## Cobertura de la rúbrica

| Criterio | Diapositivas |
|---|---|
| Contextualización y exploración | Portada, 1 |
| Metodología y justificación | 2, 3, 5 |
| Random Forest propio | 4 |
| Hiperparámetros y resultados | 6–8, respaldos A–D |
| Análisis crítico | 9 |
| Conclusiones y limitaciones | 10 |
| Claridad, tiempo y participación | 11 principales; tabla de tiempos anterior |
