"""
app.py
------
Interfaz Streamlit de la aplicación de análisis de fatiga metálica.

Ejecutar con:  streamlit run app.py
"""

import io

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import database
import diagnosis_engine
import image_analysis
import vibration_analysis

st.set_page_config(page_title="Analizador de fatiga metálica", layout="wide")
database.init_db()

with st.sidebar:
    st.markdown("### Realizado por")
    st.markdown(
        "- Alfonso Rodríguez, María José\n"
        "- Buitrago Saavedra, Julián Felipe\n"
        "- Herrera Fuentes, Luis Fernando\n"
        "- Tique Rincón, Kenny Alexander"
    )
    st.markdown("**Docente:** Efrén Eduardo Vásquez Díaz")
    st.markdown("**Universidad ECCI**")
    st.divider()

if "image_result" not in st.session_state:
    st.session_state.image_result = None
if "vibration_result" not in st.session_state:
    st.session_state.vibration_result = None

st.title("Analizador de fatiga metálica")
st.caption(
    "Proyecto de grado — simulación con datos sintéticos / datasets académicos. "
    "No usar como herramienta de decisión en un caso real sin validación adicional."
)

tab_img, tab_vib, tab_diag, tab_hist, tab_info = st.tabs(
    ["1. Imagen de fractura", "2. Señal de vibración", "3. Diagnóstico", "Historial", "Acerca de"]
)

# ---------------------------------------------------------------------------
# Tab 1: Imagen
# ---------------------------------------------------------------------------
with tab_img:
    st.subheader("Cargar y analizar la imagen de la superficie de fractura")
    uploaded_img = st.file_uploader("Imagen (jpg/png)", type=["jpg", "jpeg", "png"], key="img_uploader")

    col1, col2 = st.columns(2)
    if uploaded_img is not None:
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with col1:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Imagen original", use_container_width=True)

        if st.button("Analizar imagen"):
            result = image_analysis.analyze_image(img_bgr)
            st.session_state.image_result = result

    if st.session_state.image_result:
        result = st.session_state.image_result
        with col2:
            st.image(result["preprocessed_image"], caption="Preprocesada (gris + CLAHE)",
                      use_container_width=True, clamp=True)
        st.success(f"Clase predicha: **{result['predicted_class']}** "
                   f"(confianza: {result['confidence']:.1%})")
        st.bar_chart(pd.Series(result["probabilities"], name="probabilidad"))
        with st.expander("Ver características de textura extraídas (GLCM)"):
            st.json(result["features"])

# ---------------------------------------------------------------------------
# Tab 2: Vibración
# ---------------------------------------------------------------------------
with tab_vib:
    st.subheader("Cargar o simular la señal de vibración")
    source = st.radio("Fuente de la señal", ["Generar señal de demostración", "Cargar archivo CSV"], horizontal=True)

    signal, fs, shaft_freq = None, 12_000.0, 30.0

    if source == "Generar señal de demostración":
        fault_choice = st.selectbox(
            "Tipo de condición a simular",
            vibration_analysis.VIBRATION_CLASSES,
            format_func=lambda c: database.get_vibration_fault(c)["name"],
        )
        shaft_freq = st.slider("Frecuencia de giro del eje (Hz)", 10.0, 60.0, 30.0)
        if st.button("Generar señal"):
            signal = vibration_analysis.generate_demo_signal(fault_choice, fs=fs, shaft_freq=shaft_freq)
            st.session_state["raw_signal"] = (signal, fs, shaft_freq)
    else:
        st.caption("El CSV debe tener una sola columna con los valores de la señal en el tiempo.")
        uploaded_csv = st.file_uploader("Archivo CSV", type=["csv"], key="csv_uploader")
        fs = st.number_input("Frecuencia de muestreo (Hz)", value=12_000.0, step=100.0)
        shaft_freq = st.number_input("Frecuencia de giro del eje (Hz)", value=30.0, step=1.0)
        if uploaded_csv is not None:
            df = pd.read_csv(uploaded_csv, header=None)
            signal = df.iloc[:, 0].to_numpy(dtype=float)
            st.session_state["raw_signal"] = (signal, fs, shaft_freq)

    if "raw_signal" in st.session_state:
        signal, fs, shaft_freq = st.session_state["raw_signal"]
        st.line_chart(pd.Series(signal[:2000], name="señal (primeras 2000 muestras)"))

        if st.button("Analizar señal"):
            result = vibration_analysis.analyze_vibration(signal, fs, shaft_freq)
            st.session_state.vibration_result = result

    if st.session_state.vibration_result:
        result = st.session_state.vibration_result
        st.success(f"Clase predicha: **{result['predicted_class']}** "
                   f"(confianza: {result['confidence']:.1%})")
        st.bar_chart(pd.Series(result["probabilities"], name="probabilidad"))

        spectrum_df = pd.DataFrame({"frecuencia_hz": result["freqs"], "amplitud": result["spectrum"]})
        spectrum_df = spectrum_df[spectrum_df["frecuencia_hz"] <= shaft_freq * 10]
        st.line_chart(spectrum_df.set_index("frecuencia_hz"))
        with st.expander("Ver características espectrales extraídas"):
            st.json(result["features"])

# ---------------------------------------------------------------------------
# Tab 3: Diagnóstico combinado
# ---------------------------------------------------------------------------
with tab_diag:
    st.subheader("Diagnóstico combinado")

    img_ok = st.session_state.image_result is not None
    vib_ok = st.session_state.vibration_result is not None

    if not (img_ok and vib_ok):
        st.info("Analiza primero una imagen (pestaña 1) y una señal de vibración (pestaña 2) "
                "para generar el diagnóstico combinado.")
    else:
        img_res = st.session_state.image_result
        vib_res = st.session_state.vibration_result

        if img_res["predicted_class"] == "fatiga":
            st.markdown("##### Contexto operativo (para afinar el subtipo de fatiga)")
            st.caption(
                "Los subtipos de fatiga (alto ciclo, bajo ciclo, térmica, por "
                "corrosión, por fricción) casi no se distinguen por textura o "
                "vibración solas — dependen del contexto real de operación."
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                temperatura = st.selectbox("Temperatura de operación", ["normal", "alta", "baja"])
            with c2:
                ambiente = st.selectbox("Ambiente", ["normal", "corrosivo", "abrasivo"])
            with c3:
                carga = st.selectbox(
                    "Tipo de carga", ["alto_ciclo", "bajo_ciclo", "fretting"],
                    format_func=lambda c: {
                        "alto_ciclo": "Cíclica normal (alto ciclo)",
                        "bajo_ciclo": "Cíclica severa (bajo ciclo)",
                        "fretting": "Fricción por micro-movimiento (fretting)",
                    }[c],
                )
        else:
            temperatura, ambiente, carga = "normal", "normal", "alto_ciclo"

        if st.button("Generar diagnóstico", type="primary"):
            diag = diagnosis_engine.diagnose(
                img_res["predicted_class"], img_res["confidence"],
                vib_res["predicted_class"], vib_res["confidence"],
                temperatura=temperatura, ambiente=ambiente, carga=carga,
            )
            st.session_state.diagnosis = diag

        if "diagnosis" in st.session_state:
            diag = st.session_state.diagnosis
            st.metric("Confianza combinada", f"{diag.combined_confidence:.1%}")
            st.markdown(f"### Causa raíz probable\n{diag.root_cause}")
            st.markdown(f"**Explicación:** {diag.explanation}")
            st.markdown(f"**Recomendación:** {diag.recommendation}")
            if not diag.rule_found:
                st.warning("Esta combinación no tiene una regla específica en la base de "
                           "conocimiento; el resultado es una inferencia general.")

            if st.button("Guardar en el historial"):
                database.save_analysis(
                    diag.image_class, diag.image_confidence,
                    diag.vibration_class, diag.vibration_confidence,
                    diag.root_cause, diag.explanation,
                )
                st.success("Análisis guardado en el historial.")

# ---------------------------------------------------------------------------
# Tab 4: Historial
# ---------------------------------------------------------------------------
with tab_hist:
    st.subheader("Historial de análisis guardados")
    history = database.get_history()
    if not history:
        st.caption("Aún no hay análisis guardados.")
    else:
        st.dataframe(pd.DataFrame(history), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 5: Acerca de / base de conocimiento
# ---------------------------------------------------------------------------
with tab_info:
    st.subheader("Base de conocimiento")
    st.markdown("**Tipos de fractura**")
    conn = database.get_connection()
    st.dataframe(pd.read_sql("SELECT * FROM fracture_types", conn), use_container_width=True)
    st.markdown("**Tipos de falla por vibración**")
    st.dataframe(pd.read_sql("SELECT * FROM vibration_faults", conn), use_container_width=True)
    st.markdown("**Subtipos de fatiga (según contexto operativo)**")
    st.dataframe(pd.read_sql("SELECT * FROM fatigue_subtypes", conn), use_container_width=True)
    st.markdown("**Reglas de diagnóstico**")
    st.dataframe(pd.read_sql("SELECT * FROM diagnostic_rules", conn), use_container_width=True)
    conn.close()

    st.divider()
    st.markdown(
        "Los modelos de imagen y vibración se entrenan con datos sintéticos "
        "generados a partir de patrones descritos en la literatura de "
        "fractografía y diagnóstico de rodamientos (ver `image_analysis.py` "
        "y `vibration_analysis.py`). Para una validación experimental real, "
        "reemplaza los generadores sintéticos por datos reales etiquetados "
        "(por ejemplo, un subconjunto de un dataset público de fractografía "
        "SEM y el dataset CWRU de vibración de rodamientos) y vuelve a "
        "entrenar los modelos."
    )

    st.divider()
    st.markdown(
        "**Realizado por:** Alfonso Rodríguez, María José · "
        "Buitrago Saavedra, Julián Felipe · Herrera Fuentes, Luis Fernando · "
        "Tique Rincón, Kenny Alexander  \n"
        "**Docente:** Efrén Eduardo Vásquez Díaz  \n"
        "**Universidad ECCI**"
    )
