import streamlit as st
import pandas as pd
import os
import ast

# Función para cargar archivos según su extensión
def load_file(file_path):
    ext = file_path.split('.')[-1]
    if ext == 'csv':
        return pd.read_csv(file_path)
    elif ext == 'txt':
        return pd.read_csv(file_path, delimiter='\t')
    elif ext == 'xlsx':
        return pd.read_excel(file_path)
    else:
        return None

st.title("Buscador")

# Imprimir el directorio de trabajo actual
cwd = os.getcwd()
data_folder = os.path.join(cwd, 'web', 'pages', 'data')

# Obtener lista de archivos en la carpeta de datos
files = os.listdir(data_folder)
files = [file for file in files if file.endswith(('csv', 'xlsx'))]

# Sidebar para seleccionar el archivo
st.header("Seleccione el archivo")
selected_file = st.selectbox("Archivos disponibles", files)

if selected_file:
    file_path = os.path.join(data_folder, selected_file)
    data = load_file(file_path)
    if data is not None:
        # Buscador por nombre de columna
        search_option = st.radio("Buscar por", ('Índice', 'Nombre de columna'))

        if search_option == 'Índice':
            index = st.number_input("Ingrese el índice", min_value=0, max_value=len(data)-1, step=1)
            if "act" not in st.session_state:
                st.session_state.act = 0

            if st.button("Buscar"):
                st.write(data.iloc[index])

                if 'Images_URL' in data.columns:
                    st.markdown("**Imágenes:**")
                    img_list = ast.literal_eval(data.loc[index, 'Images_URL'])

                    # Mostrar la imagen actual
                    st.image(img_list[st.session_state["act"]].strip(), caption=f"Imagen {st.session_state['act'] + 1} de {len(img_list)}")

                    # Añadir flechas para navegar entre las imágenes
                    cols = st.columns(2)  # 2 columnas para las flechas

                    # Flecha izquierda para retroceder
                    with cols[0]:
                        if st.button("←"):
                            st.session_state["act"] = (st.session_state["act"] - 1) % len(img_list)
                    
                    # Flecha derecha para avanzar
                    with cols[1]:
                        if st.button("→"):
                            st.session_state["act"] = (st.session_state["act"] + 1) % len(img_list)

        else:
            column = st.selectbox("Seleccione la columna", data.columns)
            query = st.text_input(f"Ingrese el valor para buscar en la columna {column}")
            if st.button("Buscar"):
                st.write(data[data[column].astype(str).str.contains(query, case=False, na=False)])

    else:
        st.error("No se pudo cargar el archivo. Formato no soportado.")
