# FinIA-Flex

Copiloto Financiero y de Control de Costos para manufactura — prototipo académico.

**Caso Práctico Unidad 1**, materia Generative IA, Maestría en Ciencia de Datos y Analítica
Visual, Instituto Europeo de Posgrado.

## Qué hace

Genera reportes ejecutivos de variación presupuestal (presupuesto vs. real) para centros de
costo de una manufacturera simulada, apoyándose en:

- **RAG** sobre 4 documentos de políticas internas simuladas (`rag_docs/`).
- **Prompting** con role prompting, few-shot y chain-of-thought guiado.
- **Cálculo determinístico** de umbrales, clasificación y variación sostenida (en Python, no
  en el LLM).
- **Filtro de calidad automático** que audita cada reporte antes de mostrarlo.

Este repositorio corresponde al Paso 8 del desarrollo del prototipo. Los notebooks de Google
Colab con el desarrollo paso a paso (RAG, prompting, fine-tuning, filtros de calidad) están
documentados por separado como evidencia del proceso.

## Estructura del repositorio

```
finia_flex_app/
├── app.py                          # Aplicación Streamlit
├── requirements.txt                # Dependencias
├── rag_docs/                       # Documentos de políticas (fuente del RAG)
│   ├── 01_Politica_Gasto_Aprobaciones.md
│   ├── 02_Principios_Costeo_Variaciones.md
│   ├── 03_Formato_Reporte_Ejecutivo.md
│   └── 04_Buenas_Practicas_Optimizacion_Costos.md
├── FlexParts_Dataset_Simulado.xlsx # Dataset de ejemplo (240 filas, 3 escenarios)
└── README.md
```

## Cómo desplegarlo (gratis) en Streamlit Community Cloud

1. Crea un repositorio en GitHub y sube todo el contenido de esta carpeta (manteniendo la
   estructura, incluida la carpeta `rag_docs/`).
2. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de
   GitHub.
3. Clic en **"New app"**, selecciona el repositorio, la rama y el archivo principal
   (`app.py`).
4. Antes de desplegar, ve a **"Advanced settings" → "Secrets"** y agrega:
   ```
   GROQ_API_KEY = "tu_api_key_de_groq"
   ```
   (Esto es opcional — la app también permite ingresar la API key manualmente desde la
   barra lateral, así que no es obligatorio configurarlo aquí si prefieres no exponerlo en
   Secrets.)
5. Clic en **"Deploy"**. La primera carga tarda unos minutos porque construye el índice
   vectorial la primera vez; las siguientes cargas son rápidas gracias al caché
   (`@st.cache_resource`).

## Cómo correrlo localmente (opcional, para probar antes de desplegar)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Datos y políticas

Tanto el dataset financiero como los documentos de política son **100% simulados**, creados
específicamente para este prototipo académico — no representan datos reales de ninguna
empresa.
