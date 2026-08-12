"""
FinIA-Flex — Copiloto Financiero y de Control de Costos para manufactura
Caso Práctico Unidad 1, materia Generative IA — Maestría en Ciencia de Datos y
Analítica Visual, Instituto Europeo de Posgrado.

Interfaz web (Streamlit) que integra:
- RAG sobre políticas internas (LangChain + ChromaDB)
- Prompt maestro con grounding (Groq / Llama 3.3 70B)
- Cálculo determinístico de indicadores (sin depender del LLM para aritmética)
- Filtro de calidad automático (validar_reporte)
"""

import os
import re
import glob

import streamlit as st
import pandas as pd

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq


# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================
st.set_page_config(page_title="FinIA-Flex", page_icon="📊", layout="wide")

RAG_DOCS_PATH = os.path.join(os.path.dirname(__file__), "rag_docs")

UMBRALES_APROBACION_MXN = {
    "Materia Prima": 80000,
    "Mano de Obra Directa": 40000,
    "Energía": 30000,
    "Mantenimiento": 50000,
    "Logística/Fletes": 35000,
}

SYSTEM_PROMPT = """Eres un analista financiero senior de FlexParts Manufacturing MX,
especializado en control de costos de manufactura. Tu tarea es generar reportes ejecutivos
de variación presupuestal para Gerencia, a partir de datos de presupuesto vs. gasto real y
del contexto de políticas internas que se te proporcione.

RAZONAMIENTO INTERNO (no lo muestres en la respuesta final, solo úsalo para pensar):
1. Los indicadores de clasificación, umbral y variación sostenida ya vienen calculados por
   el sistema en el bloque "INDICADORES CALCULADOS POR EL SISTEMA". NO los recalcules, NO los
   contradigas, NO compares tú mismo montos contra umbrales — usa esos valores tal cual.
2. Con base en esos indicadores ya calculados, verifica si el contexto de políticas
   proporcionado incluye una regla que corresponda a lo que los indicadores señalan (umbral
   excedido, variación sostenida, o ninguno). Si el contexto no incluye una política aplicable
   a lo que indican los indicadores, dilo explícitamente — nunca inventes un umbral o una
   regla que no esté en el contexto.
3. Distingue causas internas (atendibles por el responsable del centro de costo) de causas
   externas (fuera de su control), cuando el contexto lo permita.
4. Si el usuario proporciona un comentario de retroalimentación, incorpóralo DENTRO de la
   sección existente que corresponda (normalmente "4. Recomendación", o "2. Diagnóstico por
   Centro de Costo" si pide más detalle sobre la causa) — como una viñeta o frase adicional
   en esa misma sección. NUNCA agregues una sección nueva, un sexto punto, ni un encabezado
   como "Recomendación Adicional". La respuesta con retroalimentación debe seguir teniendo
   exactamente las mismas 5 secciones numeradas 1 a 5, ni una más.
5. Solo después de este análisis, redacta el reporte final.

ESTRUCTURA OBLIGATORIA DE LA RESPUESTA FINAL — EXACTAMENTE 5 SECCIONES, SIEMPRE, INCLUSO
CON RETROALIMENTACIÓN DEL USUARIO:
1. Resumen Ejecutivo (máximo 3 líneas)
2. Diagnóstico por Centro de Costo (variación en MXN y %, clasificación, causa probable)
3. Alertas de Política (solo si el contexto proporcionado activa alguna; si no, escribir
   "Sin alertas de política en el contexto disponible")
4. Recomendación (una acción concreta y accionable por cada hallazgo relevante)
5. Responsable y Siguiente Paso

REGLAS DE GROUNDING:
- Usa exclusivamente los datos numéricos y el contexto de políticas que se te proporcionen.
- Si citas una política o un umbral, debe provenir textualmente del contexto recibido.
- Si el contexto no cubre algo que sería útil mencionar, indica la limitación en vez de
  completar con supuestos.
- NUNCA inventes nombres de personas, cargos o responsables. El campo "Responsable" debe
  llenarse únicamente con el valor recibido en los DATOS de entrada, copiado tal cual. Si el
  campo Responsable no viene incluido en los DATOS, escribe exactamente "No especificado en
  los datos proporcionados" — no propongas un nombre, cargo o departamento por tu cuenta bajo
  ninguna circunstancia.
- NUNCA recalcules ni contradigas los valores del bloque "INDICADORES CALCULADOS POR EL
  SISTEMA". Si ese bloque indica que el umbral NO se excedió, no afirmes lo contrario aunque
  el monto te parezca alto; si indica que sí se excedió, no lo minimices.

TONO: profesional, directo, sin tecnicismos innecesarios. El reporte debe ser legible para
un Gerente de Planta que no es especialista financiero. Evita juicios de valor sobre las
personas; evalúa procesos y resultados.
"""

EJEMPLO_FEW_SHOT_ENTRADA = """
DATOS:
Centro de costo: Línea de Producción 2
Categoría: Mantenimiento
Presupuesto: $70,000 MXN | Real: $87,200 MXN (Septiembre)
Histórico: Julio +18.6%, Agosto +20.0%, Septiembre +24.6%
Responsable: Coordinador de Mantenimiento - J. Salinas

CONTEXTO DE POLÍTICAS RECUPERADO:
"Cuando una categoría de gasto en un mismo centro de costo presenta una variación positiva
(sobrecosto) durante 3 meses consecutivos o más, el responsable debe presentar un plan
correctivo formal a Gerencia, independientemente de si cada mes individual superó o no el
umbral de aprobación." (Política POL-FIN-001, sección 4)
"""

EJEMPLO_FEW_SHOT_SALIDA = """
1. Resumen Ejecutivo
Mantenimiento en Línea de Producción 2 muestra sobrecosto sostenido por tercer mes
consecutivo, activando la regla de variación sostenida de la Política POL-FIN-001.

2. Diagnóstico por Centro de Costo
- Línea de Producción 2 / Mantenimiento: variación de +$17,200 MXN (+24.6%) en septiembre.
  Clasificación: significativa. Tendencia sostenida desde julio (+18.6%, +20.0%, +24.6%).

3. Alertas de Política
Se activa la regla de variación sostenida (POL-FIN-001, sección 4): 3 meses consecutivos de
sobrecosto en la misma categoría y centro de costo requieren plan correctivo formal a
Gerencia, independientemente del monto individual de cada mes.

4. Recomendación
Solicitar al Coordinador de Mantenimiento un plan correctivo formal antes del cierre del
siguiente mes, desagregando el gasto entre mantenimiento correctivo y preventivo para
identificar si el sobrecosto responde a fallas puntuales o a un patrón estructural.

5. Responsable y Siguiente Paso
Responsable: Coordinador de Mantenimiento - J. Salinas.
Siguiente paso: presentar plan correctivo formal a Gerencia — fecha límite sugerida: cierre
del mes en curso.
"""


# ============================================================================
# CÁLCULO DETERMINÍSTICO DE INDICADORES (Paso 5)
# ============================================================================
def clasificar_variacion(variacion_pct: float) -> str:
    if variacion_pct >= 10:
        return "Variación significativa"
    elif variacion_pct >= 3:
        return "Variación moderada"
    elif variacion_pct >= -2.9:
        return "Dentro de rango normal"
    elif variacion_pct >= -10:
        return "Ahorro saludable"
    else:
        return "Ahorro atípico"


def calcular_racha_sobrecosto(variacion_pct_actual: float, historico_mensual: list) -> int:
    racha = 1 if variacion_pct_actual > 0 else 0
    if racha == 0:
        return 0
    for _, pct in reversed(historico_mensual):
        if pct > 0:
            racha += 1
        else:
            break
    return racha


def calcular_indicadores(categoria, variacion_mxn, variacion_pct, historico_mensual=None):
    historico_mensual = historico_mensual or []
    umbral = UMBRALES_APROBACION_MXN.get(categoria)
    excede_umbral = (umbral is not None) and (variacion_mxn > umbral)
    racha = calcular_racha_sobrecosto(variacion_pct, historico_mensual)
    return {
        "clasificacion": clasificar_variacion(variacion_pct),
        "umbral_categoria_mxn": umbral,
        "excede_umbral": excede_umbral,
        "racha_meses_sobrecosto": racha,
        "variacion_sostenida": racha >= 3,
    }


# ============================================================================
# FILTRO DE CALIDAD AUTOMÁTICO (Paso 7)
# ============================================================================
def _oraciones_con_excede(texto: str) -> list:
    resultados = []
    for m in re.finditer(r"exced\w*", texto, re.IGNORECASE):
        inicio = m.start()
        antes = texto[max(0, inicio - 15):inicio].lower()
        if "no " in antes or "sin " in antes:
            continue
        ini_oracion = texto.rfind(".", 0, inicio) + 1
        fin_oracion = texto.find(".", inicio)
        fin_oracion = fin_oracion if fin_oracion != -1 else len(texto)
        resultados.append(texto[ini_oracion:fin_oracion + 1])
    return resultados


def afirma_exceso_umbral(texto: str, umbral_mxn) -> bool:
    for oracion in _oraciones_con_excede(texto):
        oracion_low = oracion.lower()
        if "umbral" in oracion_low:
            return True
        montos = [float(x.replace(",", "")) for x in re.findall(r"\$([\d,]+(?:\.\d+)?)", oracion)]
        if umbral_mxn and any(abs(m - umbral_mxn) < 1 for m in montos):
            return True
    return False


def validar_reporte(texto_reporte, responsable_esperado, variacion_mxn, indicadores, contexto_recuperado):
    problemas = []

    secciones_esperadas = [
        "1. Resumen Ejecutivo", "2. Diagnóstico por Centro de Costo",
        "3. Alertas de Política", "4. Recomendación", "5. Responsable y Siguiente Paso",
    ]
    faltantes = [s for s in secciones_esperadas if s not in texto_reporte]
    if faltantes:
        problemas.append(f"Estructura incompleta. Faltan secciones: {faltantes}")

    # Detecta secciones numeradas extra (6, 7, ...) o encabezados no estándar tipo
    # "Recomendación Adicional" que a veces aparecen al incorporar retroalimentación
    seccion_extra = re.search(r"(?:^|\n)\s*6\.\s+\S", texto_reporte) or \
        re.search(r"[Rr]ecomendaci[oó]n\s+[Aa]dicional", texto_reporte)
    if seccion_extra:
        problemas.append(
            "El reporte parece incluir una sección extra fuera de la estructura de 5 "
            "secciones obligatorias (posible sección 6 o 'Recomendación Adicional')."
        )

    match_resp = re.search(r"Responsable:\s*(.+?)(?:\n|$)", texto_reporte)
    citado = match_resp.group(1).strip() if match_resp else None
    if citado is None:
        problemas.append("No se encontró el campo Responsable en el texto generado.")
    elif responsable_esperado.lower() not in citado.lower() and citado.lower() not in responsable_esperado.lower():
        problemas.append(f'Responsable inconsistente: el reporte dice "{citado}", pero el dato de entrada era "{responsable_esperado}".')

    variacion_abs = abs(variacion_mxn)
    montos_en_texto = [float(m.replace(",", "")) for m in re.findall(r"\$([\d,]+(?:\.\d+)?)", texto_reporte)]
    if not any(abs(m - variacion_abs) < 1 for m in montos_en_texto):
        problemas.append(f"La variación calculada (${variacion_abs:,.0f} MXN) no aparece explícitamente en el texto generado.")

    afirma_excede = afirma_exceso_umbral(texto_reporte, indicadores["umbral_categoria_mxn"])
    if afirma_excede and not indicadores["excede_umbral"]:
        problemas.append("El texto afirma que se excede el umbral, pero el indicador calculado dice que NO — inconsistencia numérica grave.")
    if indicadores["excede_umbral"] and not afirma_excede:
        problemas.append("El indicador señala que SÍ se excede el umbral, pero el texto no lo menciona.")

    afirma_sostenida = bool(re.search(r"variaci[oó]n\s+sostenida", texto_reporte, re.IGNORECASE))
    if indicadores["variacion_sostenida"] and not afirma_sostenida:
        problemas.append("El indicador señala variación sostenida, pero el texto no la menciona.")

    codigos_citados = set(re.findall(r"POL-FIN-\d{3}", texto_reporte))
    codigos_contexto = set(re.findall(r"POL-FIN-\d{3}", contexto_recuperado))
    no_respaldados = codigos_citados - codigos_contexto
    if no_respaldados:
        problemas.append(f"Códigos de política citados sin respaldo en el contexto: {no_respaldados}")

    return {"aprobado": len(problemas) == 0, "problemas": problemas}


# ============================================================================
# RECURSOS COMPARTIDOS (cacheados — se cargan una sola vez)
# ============================================================================
@st.cache_resource(show_spinner="Cargando base de conocimiento de políticas...")
def cargar_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    loader = DirectoryLoader(RAG_DOCS_PATH, glob="*.md", loader_cls=TextLoader,
                              loader_kwargs={"encoding": "utf-8"})
    documentos = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    fragmentos = splitter.split_documents(documentos)
    return Chroma.from_documents(documents=fragmentos, embedding=embeddings)


def recuperar_contexto(vectorstore, centro_costo, categoria, resumen_situacion, historico="", k=3):
    consulta = (
        f"Política de aprobación y control de gasto para la categoría {categoria} "
        f"en el centro de costo {centro_costo}. Situación actual: {resumen_situacion}. "
        f"Histórico reciente: {historico}. "
        f"¿Qué umbral de aprobación o regla de variación sostenida aplica?"
    )
    resultados = vectorstore.similarity_search(consulta, k=k)
    fragmentos_texto = []
    for r in resultados:
        fuente = r.metadata["source"].split("/")[-1].split("\\")[-1]
        fragmentos_texto.append(f'({fuente})\n"{r.page_content}"')
    return "\n\n".join(fragmentos_texto)


def generar_reporte(client, vectorstore, centro_costo, categoria, presupuesto, real,
                     historico_texto, historico_mensual, responsable, comentario=None):
    variacion_mxn = real - presupuesto
    variacion_pct = (variacion_mxn / presupuesto * 100) if presupuesto else 0
    indicadores = calcular_indicadores(categoria, variacion_mxn, variacion_pct, historico_mensual)

    resumen_situacion = f"variación de {variacion_pct:.1f}% ({variacion_mxn:,.0f} MXN)"
    contexto = recuperar_contexto(vectorstore, centro_costo, categoria, resumen_situacion, historico=historico_texto)

    responsable_texto = responsable if responsable else "No especificado en los datos proporcionados"
    umbral_texto = (f"${indicadores['umbral_categoria_mxn']:,.0f} MXN"
                     if indicadores["umbral_categoria_mxn"] else "sin umbral definido para esta categoría")

    caso_entrada = f"""
DATOS:
Centro de costo: {centro_costo}
Categoría: {categoria}
Presupuesto: ${presupuesto:,.0f} MXN | Real: ${real:,.0f} MXN
Variación: {variacion_pct:.1f}% (${variacion_mxn:,.0f} MXN)
Histórico: {historico_texto}
Responsable: {responsable_texto}

INDICADORES CALCULADOS POR EL SISTEMA (usa estos valores tal cual, no los recalcules ni los
contradigas — fueron calculados por código, no por ti):
- Clasificación de la variación: {indicadores['clasificacion']}
- Umbral de aprobación de la categoría "{categoria}": {umbral_texto}
- ¿La variación de este mes excede el umbral de su categoría?: {"SÍ" if indicadores['excede_umbral'] else "NO"}
- Meses consecutivos de sobrecosto (incluyendo el actual): {indicadores['racha_meses_sobrecosto']}
- ¿Aplica la regla de variación sostenida (3+ meses consecutivos de sobrecosto)?: {"SÍ" if indicadores['variacion_sostenida'] else "NO"}

CONTEXTO DE POLÍTICAS RECUPERADO (automático):
{contexto}
"""
    if comentario:
        caso_entrada += f"\nCOMENTARIO DEL USUARIO PARA AJUSTAR EL REPORTE:\n{comentario}\n"

    respuesta = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EJEMPLO_FEW_SHOT_ENTRADA},
            {"role": "assistant", "content": EJEMPLO_FEW_SHOT_SALIDA},
            {"role": "user", "content": caso_entrada},
        ],
        temperature=0.3,
    )
    texto_reporte = respuesta.choices[0].message.content
    validacion = validar_reporte(texto_reporte, responsable_texto, variacion_mxn, indicadores, contexto)
    return texto_reporte, validacion, indicadores


# ============================================================================
# INTERFAZ
# ============================================================================
st.title("📊 FinIA-Flex")
st.caption("Copiloto Financiero y de Control de Costos — FlexParts Manufacturing MX (prototipo académico)")

with st.sidebar:
    st.header("Configuración")
    groq_api_key = st.text_input(
        "API key de Groq", type="password",
        help="Obtén una gratis en console.groq.com/keys. No se guarda en ningún lado.",
    )
    st.divider()
    st.subheader("Datos financieros")
    archivo = st.file_uploader("Sube tu archivo Excel (opcional)", type=["xlsx"],
                                help="Debe tener columnas: Mes, Mes #, Centro de Costo, "
                                     "Categoría de Gasto, Presupuesto (MXN), Real (MXN), Responsable")

if archivo is not None:
    df = pd.read_excel(archivo, sheet_name="Datos")
else:
    st.sidebar.info("No se subió archivo — usando el dataset de ejemplo de FlexParts.")
    ruta_ejemplo = os.path.join(os.path.dirname(__file__), "FlexParts_Dataset_Simulado.xlsx")
    df = pd.read_excel(ruta_ejemplo, sheet_name="Datos") if os.path.exists(ruta_ejemplo) else None

if df is None:
    st.warning("No hay datos disponibles. Sube un archivo Excel en la barra lateral.")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    centro_costo_sel = st.selectbox("Centro de costo", sorted(df["Centro de Costo"].unique()))
with col2:
    categorias_disp = sorted(df[df["Centro de Costo"] == centro_costo_sel]["Categoría de Gasto"].unique())
    categoria_sel = st.selectbox("Categoría de gasto", categorias_disp)
with col3:
    subset = df[(df["Centro de Costo"] == centro_costo_sel) & (df["Categoría de Gasto"] == categoria_sel)].sort_values("Mes #")
    mes_sel = st.selectbox("Mes a analizar", subset["Mes"].tolist(), index=len(subset) - 1 if len(subset) else 0)

fila_actual = subset[subset["Mes"] == mes_sel].iloc[0]
idx_actual = subset.index.get_loc(fila_actual.name)
historico_previo = subset.iloc[:idx_actual]  # meses anteriores al seleccionado, en orden

presupuesto = float(fila_actual["Presupuesto (MXN)"])
real = float(fila_actual["Real (MXN)"])
responsable = str(fila_actual.get("Responsable", "")) or None
variacion_pct_actual = (real - presupuesto) / presupuesto * 100 if presupuesto else 0

historico_mensual = [
    (row["Mes"], (row["Real (MXN)"] - row["Presupuesto (MXN)"]) / row["Presupuesto (MXN)"] * 100)
    for _, row in historico_previo.tail(6).iterrows()
]
historico_texto = ", ".join(f"{mes} {pct:+.1f}%" for mes, pct in historico_mensual) or "sin histórico previo disponible"

st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Presupuesto", f"${presupuesto:,.0f} MXN")
m2.metric("Real", f"${real:,.0f} MXN")
m3.metric("Variación", f"{variacion_pct_actual:+.1f}%", delta=f"${real - presupuesto:,.0f} MXN")

with st.expander("Ver histórico usado para este análisis"):
    st.write(historico_texto)
    st.caption(f"Responsable registrado: {responsable or 'No especificado en los datos'}")

generar = st.button("🔍 Generar Reporte Ejecutivo", type="primary", use_container_width=True)

if "reporte_actual" not in st.session_state:
    st.session_state.reporte_actual = None
    st.session_state.validacion_actual = None

if generar:
    if not groq_api_key:
        st.error("Ingresa tu API key de Groq en la barra lateral para continuar.")
    else:
        client = Groq(api_key=groq_api_key)
        vectorstore = cargar_vectorstore()
        with st.spinner("Buscando políticas relevantes y generando el reporte..."):
            texto, validacion, indicadores = generar_reporte(
                client, vectorstore, centro_costo_sel, categoria_sel, presupuesto, real,
                historico_texto, historico_mensual, responsable,
            )
        st.session_state.reporte_actual = texto
        st.session_state.validacion_actual = validacion
        st.session_state.datos_ultimo_caso = dict(
            centro_costo=centro_costo_sel, categoria=categoria_sel, presupuesto=presupuesto,
            real=real, historico_texto=historico_texto, historico_mensual=historico_mensual,
            responsable=responsable,
        )

if st.session_state.reporte_actual:
    st.divider()
    validacion = st.session_state.validacion_actual
    if validacion["aprobado"]:
        st.success("✅ Reporte aprobado por el filtro de calidad automático")
    else:
        st.warning("⚠️ El filtro de calidad detectó posibles inconsistencias — revisar antes de usar:")
        for p in validacion["problemas"]:
            st.write(f"- {p}")

    st.markdown(st.session_state.reporte_actual)

    st.divider()
    st.subheader("💬 Retroalimentación")
    comentario = st.text_area(
        "¿Qué ajustarías de este reporte? (opcional)",
        placeholder="Ej. 'sé más específico sobre la causa probable' o 'agrega una alternativa de recomendación'",
    )
    if st.button("Regenerar con este comentario") and comentario and groq_api_key:
        client = Groq(api_key=groq_api_key)
        vectorstore = cargar_vectorstore()
        datos = st.session_state.datos_ultimo_caso
        with st.spinner("Regenerando el reporte con tu comentario..."):
            texto, validacion, _ = generar_reporte(
                client, vectorstore, datos["centro_costo"], datos["categoria"],
                datos["presupuesto"], datos["real"], datos["historico_texto"],
                datos["historico_mensual"], datos["responsable"], comentario=comentario,
            )
        st.session_state.reporte_actual = texto
        st.session_state.validacion_actual = validacion
        st.rerun()

st.divider()
st.caption(
    "FinIA-Flex — Prototipo académico, Caso Práctico Unidad 1, materia Generative IA, "
    "Maestría en Ciencia de Datos y Analítica Visual, Instituto Europeo de Posgrado. "
    "Dataset y políticas 100% simulados."
)
