import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

# AGENTE INTELIGENTE PROBABILÍSTICO
# Detección de sospecha de Dengue vs COVID-19

st.set_page_config(
    page_title="Agente Inteligente Dengue vs COVID",
    layout="centered"
)

st.title("Agente Inteligente Probabilístico")
st.subheader("Detección de sospecha de Dengue vs COVID-19")
st.write(
    "Este sistema utiliza Naive Bayes para estimar la probabilidad de que un paciente "
    "presente sospecha de Dengue o COVID-19 según sus síntomas."
)


@st.cache_data
def cargar_datos():
    return pd.read_csv("dataset_dengue_covid.csv")

df = cargar_datos()

X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

modelo = CategoricalNB()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# st.sidebar.header("Información del modelo")
# st.sidebar.write(f"Registros totales: {len(df)}")
# st.sidebar.write(f"Precisión del modelo: {accuracy:.2%}")
# st.sidebar.write("Algoritmo: Categorical Naive Bayes")


st.markdown("---")
st.header("Ingrese los síntomas del paciente")


def opcion_binaria(nombre):
    return 1 if st.checkbox(nombre) else 0


fever = opcion_binaria("Fiebre")
headache = opcion_binaria("Dolor de cabeza")
joint_pain = opcion_binaria("Dolor articular / muscular")
bleeding = opcion_binaria("Sangrado leve")
dry_cough = opcion_binaria("Tos seca")
breathing_difficulty = opcion_binaria("Dificultad respiratoria")
sore_throat = opcion_binaria("Dolor de garganta")
diarrhea = opcion_binaria("Diarrea")


if st.button("Analizar paciente"):
    if sum([
        fever,
        headache,
        joint_pain,
        bleeding,
        dry_cough,
        breathing_difficulty,
        sore_throat,
        diarrhea
    ]) == 0:
        st.error("Debe seleccionar al menos un síntoma.")

    else:
        sintomas = []

        if fever:
            sintomas.append("Fiebre")

        if headache:
            sintomas.append("Dolor de cabeza")

        if joint_pain:
            sintomas.append("Dolor articular / muscular")

        if bleeding:
            sintomas.append("Sangrado leve")

        if dry_cough:
            sintomas.append("Tos seca")

        if breathing_difficulty:
            sintomas.append("Dificultad respiratoria")

        if sore_throat:
            sintomas.append("Dolor de garganta")

        if diarrhea:
            sintomas.append("Diarrea")

        st.write("Síntomas detectados:")
        st.write(", ".join(sintomas))

        nuevo_paciente = pd.DataFrame([
            {
                "Fever": fever,
                "Headache": headache,
                "JointPain": joint_pain,
                "Bleeding": bleeding,
                "DryCough": dry_cough,
                "BreathingDifficulty": breathing_difficulty,
                "SoreThroat": sore_throat,
                "Diarrhea": diarrhea
            }
        ])

        prediccion = modelo.predict(nuevo_paciente)[0]
        probabilidades = modelo.predict_proba(nuevo_paciente)[0]
        clases = modelo.classes_

        st.markdown("---")
        st.header("Resultado del análisis")

        st.success(f"Diagnóstico probable: {prediccion}")

        st.subheader("Probabilidades estimadas")

        for clase, prob in zip(clases, probabilidades):
            st.write(f"{clase}: {prob:.2%}")
            st.progress(float(prob))

        if prediccion == "Dengue":
            st.warning(
                "El patrón de síntomas presenta mayor compatibilidad con Dengue. "
                "Se recomienda evaluación médica y control clínico."
            )
        else:
            st.warning(
                "El patrón de síntomas presenta mayor compatibilidad con COVID-19. "
                "Se recomienda evaluación médica y seguimiento respiratorio."
            )

        st.info(
            "Este resultado representa una sospecha probabilística y no reemplaza "
            "el diagnóstico médico profesional."
        )

st.markdown("---")
st.caption("Clasificación probabilística de Dengue vs COVID-19 usando Naive Bayes")
