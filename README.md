# TechCareer

Proyecto de predicción de carrera para perfiles tech:
- Modelo 1: clasificar CV → rol tech.
- Modelo 2: predecir seniority y salario a partir del rol + contexto.

Este README explica solo la estructura de carpetas (sin frontend aún).

---

## Estructura del proyecto

```text
techcareer/
 ├─ data/
 │   ├─ model1_cv_role/
 │   │   ├─ raw/
 │   │   ├─ cleaning/
 │   │   └─ processed/
 │   └─ model2_salaries/
 │       ├─ raw/
 │       ├─ cleaning/
 │       └─ processed/
 ├─ models/
 │   ├─ cv_role/
 │   └─ salaries/
 └─ src/
     ├─ model1_cv_role_training/
     └─ model2_salaries_training/
```
## `data/`

Contiene datasets para cada modelo.

---

### `data/model1_cv_role/`

Datos para el **Modelo 1: CV → rol tech**.

#### `data/model1_cv_role/raw/`
Datasets originales tal cual se descargan (Kaggle, etc.).

Ejemplos de ficheros:

- `resume_dataset_avishek.csv`
- `resume_dataset_jithin.csv`

#### `data/model1_cv_role/cleaning/`
Ficheros intermedios durante la limpieza y unificación.

Ejemplos:

- CSVs con categorías ya mapeadas a nuestros roles
- Resultados de merges, filtros, etc.

#### `data/model1_cv_role/processed/`
Datos listos para entrenar el modelo.

Ejemplo:

- `cv_labeled_final.csv`  
  - Columnas tipo: `cv_text`, `role_label_final`.

---

### `data/model2_salaries/`

Datos para el **Modelo 2: seniority + salario**.

#### `data/model2_salaries/raw/`
Datasets originales de salarios.

Ejemplos:

- `data_science_job_salaries_ruchi.csv`
- `latest_ds_job_salaries_2020_2025.csv`
- `software_professional_salary.csv`

#### `data/model2_salaries/cleaning/`
Ficheros intermedios tras mezclar y limpiar.

Ejemplos:

- `salaries_merged_raw.csv`
- `salaries_with_seniority_tmp.csv`  
  (con `experience_level` mapeado EN/MI/SE/EX → Junior/Mid/Senior)

#### `data/model2_salaries/processed/`
Datos definitivos para entrenar modelos de seniority y salario.

Ejemplos:

- `salaries_clean_for_seniority.csv`
- `salaries_clean_for_regression.csv`

---

## `models/`

Aquí guardamos los **modelos entrenados** (`.pkl`, `.joblib`).

### `models/cv_role/`
Artefactos del **Modelo 1**:

- `cv_role_pipeline.pkl`  
  *(o bien `cv_role_vectorizer.pkl` + `cv_role_classifier.pkl` si se guardan por separado)*

### `models/salaries/`
Artefactos del **Modelo 2**:

- `seniority_model.pkl`
- `salary_regressor.pkl`
- (opcional) tablas auxiliares, p. ej. `role_seniority_salary_stats.csv`

---

## `src/`

Código Python de **entrenamiento** de los modelos  
*(más adelante aquí vivirá también la parte de inferencia/API).*

### `src/model1_cv_role_training/`

Scripts / notebooks convertidos a código para:

- Cargar datos de `data/model1_cv_role/processed/`
- Entrenar el clasificador **CV → rol**
- Guardar el modelo en `models/cv_role/`

### `src/model2_salaries_training/`

Scripts para:

- Entrenar el modelo de **seniority**
- Entrenar el modelo de **salario**
- Guardar ambos en `models/salaries/`

Más adelante añadiremos:

- Módulos de **inferencia** (p. ej. `src/model1_cv_role_inference.py`)
- La **API Flask**

---

## Flujo de trabajo (resumen)

1. Descargar datasets → ponerlos en `data/.../raw/`.
2. Limpiar y unificar datos → guardar resultados en `data/.../cleaning/`.
3. Generar datasets finales para ML → `data/.../processed/`.
4. Ejecutar scripts de `src/..._training/` → modelos entrenados en `models/`.

## Entorn

1. Instal·lar uv
2. `uv sync`
3. `uv run jupyter lab`
