# ============================================================
# CABECERA
# ============================================================
# Alumno: Clara Goñi Echeverría
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)

# Importar librerias

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """Actúa como un analista de datos experto en música y Spotify.

## OBJETIVO:
Tu misión es generar código Python usando Plotly Express (px) basándote en las instrucciones del usuario.
IMPORTANTE: 
- Atiende tanto a preguntas directas como a órdenes e instrucciones imperativas (ej: "compara mi top 5...", "muestra un gráfico...", "analiza..."). No esperes necesariamente signos de interrogación para actuar.
- Todos los textos del gráfico (títulos, etiquetas de ejes) DEBEN estar en castellano.

## DATOS DISPONIBLES:
El DataFrame se llama 'df' y tiene estas columnas exactas:
- 'ts': Fecha y hora (datetime). Rango: {fecha_min} a {fecha_max}.
- 'master_metadata_track_name': Título de la canción.
- 'master_metadata_album_artist_name': Artista principal.
- 'minutes_played': Tiempo escuchado en minutos.
- 'hour': Hora del día (0-23).
- 'day_name': Nombre del día en castellano (Lunes, Martes...).
- 'season': Estación del año ({seasons}).
- 'skipped': Booleano (True si se saltó la canción, False si se escuchó).
- 'shuffle': Booleano (True si estaba activado el modo aleatorio, False si estaba en orden).
- 'platform': Dispositivo o plataforma de escucha (ej: 'android', 'ios', 'web').
- 'periodo_semana': Indica si es 'Entre semana' o 'Fin de semana'.


## REGLAS DE ORO:
1. El código DEBE crear una variable llamada 'fig'.
2. Los títulos de los gráficos deben ser descriptivos y en CASTELLANO.
3. **COLORES EN PLOTLY EXPRESS**: 
   - Si la gráfica representa una sola categoría o no requiere dividir por grupos, NO uses el parámetro `color`. Para fijar el color verde de Spotify usa ESTRICTAMENTE: `color_discrete_sequence=['#1DB954']`. Nunca pases el código hex directamente al parámetro `color`.
   - Si en la gráfica hay más de un grupo (ej: varias estaciones, días, dispositivos), DEBES usar el parámetro `color` apuntando exclusivamente al nombre de la columna que contiene los grupos (ej: `color='season'`).
4. Para ránkings, agrupa y usa .nlargest() o .sort_values().
5. **COMPARACIONES (Ej: Verano vs Invierno)**: 
   - Filtra el DataFrame por las categorías a comparar: `df[df['season'].isin(['Verano', 'Invierno'])]`.
   - Agrupa por la categoría Y el artista/canción: `.groupby(['season', 'master_metadata_album_artist_name'])`.
   - SIEMPRE usa `color='season'` en `px.bar` para garantizar un color distinto por grupo, y añade `barmode='group'` para mostrar las barras una al lado de la otra.
6. Usa el parámetro 'labels' en px para renombrar columnas en los ejes (ej: labels={{'minutes_played': 'Minutos Escuchados'}}).
7. Para órdenes sobre "entre semana", filtra df[df['periodo_semana'] == 'Entre semana'].
8. Para órdenes sobre "fines de semana", filtra df[df['periodo_semana'] == 'Fin de semana'].
9. **MANEJO DE TIEMPO (Minutos a Horas)**: 
   - Si el usuario pide explícitamente ver el TIEMPO TOTAL escuchado o DURACIÓN TOTAL ACUMULADA, DEBES crear una columna llamada 'hours_played' (`df['hours_played'] = df['minutes_played'] / 60`) y graficar esa columna.
10. **ANÁLISIS DE MODO ALEATORIO (Shuffle)**:
    - Si el usuario pregunta si escucha más en modo "shuffle" o en "orden", NO inventes datos ni uses `.sample()`.
    - Agrupa el DataFrame usando la columna booleana `'shuffle'`.
    - Recuerda aplicar la regla 9 y sumar la columna `'hours_played'` para el eje Y.
    - Mapea los valores de la columna 'shuffle' para que en la gráfica se lea 'Modo Aleatorio' (para True) y 'Modo Orden' (para False).
11. **INTERPRETACIÓN DE "SALTAR" O "SALTOS" (Skips) EN PORCENTAJE**:
    - Si el usuario pregunta explícitamente por el "porcentaje", "tasa" o "proporción" de saltos, calcula la media: `df['skipped'].mean() * 100`.
12. **DIFERENCIA ENTRE "VECES ESCUCHADA" (Frecuencia) Y "TIEMPO" (Duración)**:
    - Si el usuario pregunta por canciones o artistas más escuchados en términos de "VECES", "reproducciones" o "frecuencia", NO sumes los minutos ni las horas. En su lugar, cuenta el número de registros usando `.size()` o `.count()`.
13. **CÁLCULO DE "CANCIONES NUEVAS" O "DESCUBRIMIENTOS"**:
    - Si el usuario pregunta por canciones "nuevas", "descubiertas" o "en qué mes descubrí más música", el código debe:
      a) Obtener la primera fecha de escucha por canción: `df_first = df.groupby('master_metadata_track_name')['ts'].min().reset_index()`.
      b) Extraer el número de mes (`df_first['Num_Mes'] = df_first['ts'].dt.month`) y mapearlo a nombres en castellano (1->'Enero', 2->'Febrero', etc.).
      c) Agrupar por el nombre del mes, ordenar cronológicamente por el número del mes para que no se desordenen alfabéticamente, y contar las ocurrencias con `.size()`.
      d) SIEMPRE añade la línea `fig.update_xaxes(type='category')` justo después de crear el objeto `fig` para que no se generen semanas ni días fantasmas en el eje X.
14. **DURACIÓN DE LA CANCIÓN VS TIEMPO TOTAL**:
    - Si el usuario pregunta por la canción "más larga", la de "mayor duración" o similar, NO sumes los minutos de todas las reproducciones. Busca el valor máximo de minutos registrados para esa canción: `df.groupby('master_metadata_track_name')['minutes_played'].max().reset_index()`.
15. **CONTEO ABSOLUTO DE SALTOS (VECES que te saltaste algo)**:
    - Si el usuario pregunta por canciones o artistas saltados "más veces" o "menos veces", NO calcules porcentajes ni proporciones en el eje Y. Filtra donde `skipped == True` y cuenta las filas agrupadas con `.size()`.
16. **EVOLUCIÓN TEMPORAL (Por mes, por año, etc.)**:
    - Si el usuario pregunta por la "evolución", "histórico" o cómo cambia algo "por mes" o "a lo largo del tiempo", DEBES generar un gráfico de líneas (`px.line`). Extrae el mes y año en formato texto usando `df['ts'].dt.strftime('%Y-%m')` para evitar distorsiones en el eje X.

## FORMATO DE RESPUESTA:
Debes responder ESTRICTAMENTE con un objeto JSON que siga esta estructura:
{{
  "tipo": "grafico",
  "codigo": "...fig = px.bar(..., title='Tu Título en Castellano', labels={{...}})...",
  "interpretacion": "Tu análisis breve en castellano de lo que muestra el gráfico."
}}
"""

# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # ----------------------------------------------------------
    # >>> TU PREPARACIÓN DE DATOS ESTÁ AQUÍ <<<
    # ----------------------------------------------------------
    # Transforma el dataset para facilitar el trabajo del LLM.
    # Lo que hagas aquí determina qué columnas tendrá `df`,
    # y tu system prompt debe describir exactamente esas columnas.
    #
    # Cosas que podrías considerar:
    # - Convertir 'ts' de string a datetime
    # - Crear columnas derivadas (hora, día de la semana, mes...)
    # - Convertir milisegundos a unidades más legibles
    # - Renombrar columnas largas para simplificar el código generado
    # - Filtrar registros que no aportan al análisis (podcasts, etc.)
    # ----------------------------------------------------------
# Asegurar que la columna platform existe y está limpia
    if 'platform' not in df.columns:
        # Si por casualidad la columna tiene otro nombre en tu JSON, cámbialo aquí
        df['platform'] = "Desconocido"

    # 1. Convertir timestamp a formato fecha y hora

    df['ts'] = pd.to_datetime(df['ts'])
    
    # Mapeo de días de la semana a castellano
    dias_es = {
        'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
        'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    df['day_name'] = df['ts'].dt.day_name().map(dias_es)

    # Nueva columna para facilitar el filtrado al LLM
    df['periodo_semana'] = df['ts'].dt.dayofweek.apply(
        lambda x: 'Fin de semana' if x >= 5 else 'Entre semana'
    )
    
    # 2. Crear columnas temporales 
    df['hour'] = df['ts'].dt.hour
    df['day_name'] = df['ts'].dt.day_name()
    df['month_name'] = df['ts'].dt.month_name()
    df['is_weekend'] = df['ts'].dt.dayofweek >= 5

    # 3. Convertir milisegundos a minutos (para que el usuario lo entienda)
    df['minutes_played'] = df['ms_played'] / 60000

    # 4. Definir estaciones para preguntas de comparación (verano vs invierno)
    def get_season(month):
        if month in [12, 1, 2]: return 'Invierno'
        if month in [3, 4, 5]: return 'Primavera'
        if month in [6, 7, 8]: return 'Verano'
        return 'Otoño'
    
    df['season'] = df['ts'].dt.month.apply(get_season)

    # 5. Limpieza básica: Asegurar que skipped sea booleano
    df['skipped'] = df['skipped'].fillna(False).astype(bool)

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min().strftime('%Y-%m-%d')
    fecha_max = df["ts"].max().strftime('%Y-%m-%d')
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()
    seasons = df["season"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
        seasons=seasons
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create( 
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    try:
        # Ejecutamos el código
        exec(code, {}, local_vars)
        # Extraemos la figura
        fig = local_vars.get("fig")
        return fig
    except Exception as e:
        # ESTA LÍNEA ES LA QUE TE DIRÁ EL ERROR EN LA WEB
        st.error(f"ERROR REAL: {e}")
        return None


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error(f"ERROR REAL: {e}")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    [Tu respuesta aquí]
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    [Tu respuesta aquí]
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    [Tu respuesta aquí]