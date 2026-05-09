import pandas as pd
import numpy as np

# SCRIPT DE PREPARACIÓN PARA EL AI PROBABILÍSTICO
# Detección de Dengue - COVID-19

np.random.seed(42)

archivo_dengue = "dataset-dengue.csv"
archivo_covid = "dataset-covid19.csv"
archivo_salida = "dataset_dengue_covid.csv"

print("Cargando datasets...")

df_dengue = pd.read_csv(archivo_dengue)
df_covid = pd.read_csv(archivo_covid)

print(f"Registros Dengue: {len(df_dengue)}")
print(f"Registros COVID: {len(df_covid)}")

#Limpiamos el dataset de dengue
print("\nLimpiando dataset de Dengue...")

columnas_dengue = [
    "Fever",
    "Headache",
    "JointPain",
    "Bleeding"
]

columnas_dengue_existentes = [
    col for col in columnas_dengue
    if col in df_dengue.columns
]

df_dengue = df_dengue[columnas_dengue_existentes].copy()

#Ajustamos las probabilidades de los sintomas de covid que son muy poco frecuentes en Dengue

df_dengue["DryCough"] = np.random.choice(
    [0, 1],
    size=len(df_dengue),
    p=[0.80, 0.20]
)

df_dengue["BreathingDifficulty"] = np.random.choice(
    [0, 1],
    size=len(df_dengue),
    p=[0.95, 0.05]
)

df_dengue["SoreThroat"] = np.random.choice(
    [0, 1],
    size=len(df_dengue),
    p=[0.80, 0.20]
)

df_dengue["Diarrhea"] = np.random.choice(
    [0, 1],
    size=len(df_dengue),
    p=[0.70, 0.30]
)

df_dengue["Diagnosis"] = "Dengue"


#Limpiamos el dataset de COVID19
print("Limpiando dataset de COVID...")

mapeo_columnas_covid = {
    "Fever": "Fever",
    "Dry-Cough": "DryCough",
    "Difficulty-in-Breathing": "BreathingDifficulty",
    "Sore-Throat": "SoreThroat",
    "Diarrhea": "Diarrhea",
    "Pains": "JointPain"
}

columnas_covid_existentes = [
    col for col in mapeo_columnas_covid.keys()
    if col in df_covid.columns
]

df_covid = df_covid[columnas_covid_existentes].copy()

for original, nuevo in mapeo_columnas_covid.items():
    if original in df_covid.columns:
        df_covid.rename(columns={original: nuevo}, inplace=True)


#Ajustamos las probabilidades de sintomas de dengue que son poco frecuentes en covid19
df_covid["Headache"] = np.random.choice(
    [0, 1],
    size=len(df_covid),
    p=[0.45, 0.55]
)

df_covid["Bleeding"] = np.random.choice(
    [0, 1],
    size=len(df_covid),
    p=[0.95, 0.05]
)

df_covid["Diagnosis"] = "COVID"


#Unificamos los datasets
print("Unificando estructura de datos...")

columnas_finales = [
    "Fever",
    "Headache",
    "JointPain",
    "Bleeding",
    "DryCough",
    "BreathingDifficulty",
    "SoreThroat",
    "Diarrhea",
    "Diagnosis"
]

for col in columnas_finales:
    if col not in df_dengue.columns:
        df_dengue[col] = 0

    if col not in df_covid.columns:
        df_covid[col] = 0

df_dengue = df_dengue[columnas_finales]
df_covid = df_covid[columnas_finales]


#Reducimos el dataset a 5mil registros de cada enfermedad
print("Reduciendo tamaño del dataset...")

limite_dengue = min(10000, len(df_dengue))
limite_covid = min(10000, len(df_covid))

sample_dengue = df_dengue.sample(
    n=limite_dengue,
    random_state=42
)

sample_covid = df_covid.sample(
    n=limite_covid,
    random_state=42
)


#Generamos el dataset unificado
print("Generando dataset unificado...")

df_final = pd.concat(
    [sample_dengue, sample_covid],
    ignore_index=True
)

df_final.dropna(inplace=True)

#Mezclamos los registros entre si
df_final = df_final.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


#Exportamos el archivo
print(f"Guardando archivo final: {archivo_salida}")

df_final.to_csv(archivo_salida, index=False)


#Resumen del proceso
print("\n========== PROCESO FINALIZADO ==========")

print(f"Total registros finales: {len(df_final)}")

print("\nDistribución por diagnóstico:")
print(df_final["Diagnosis"].value_counts())

print(f"\nArchivo generado correctamente: {archivo_salida}")

print("========================================")