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

# Función para inicializar el estado de la sesión
def initialize_session_state():
    if 'index' not in st.session_state:
        st.session_state.index = 0
    if 'search_option' not in st.session_state:
        st.session_state.search_option = 'Índice'
    if 'search_value' not in st.session_state:
        st.session_state.search_value = None

st.title(f"Buscador")

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
        # Inicializar el estado de la sesión
        initialize_session_state()
        
        # Buscador por nombre de columna
        st.session_state.search_option = st.radio("Buscar por", ('Índice', 'Nombre de columna'), index=0 if st.session_state.search_option == 'Índice' else 1)
        
        if st.session_state.search_option == 'Índice':
            index = st.number_input("Ingrese el índice", min_value=0, max_value=len(data)-1, step=1)
            if st.button("Buscar"):
                st.write(data.iloc[index])
                
                # Mostrar las imágenes si están disponibles
                if 'Images_URL' in data.columns and isinstance(data.loc[index, 'Images_URL'], str):
                    st.markdown("**Imágenes:**")
                    img_list = ast.literal_eval(data.loc[index, 'Images_URL'])
                    if img_list:  # Verificar si hay imágenes en la lista
                        for i, img_url in enumerate(img_list, start=1):
                            st.image(img_url.strip(), caption=f"{i} de {len(img_list)}")
                        
                        # Añadir flechas para navegar entre las imágenes
                        cols = st.columns(2)  # 2 columnas para las flechas
                        
                        # Flecha izquierda para retroceder
                        with cols[0]:
                            if st.session_state.index > 0:
                                if st.button("←"):
                                    st.session_state.index -= 1
                        
                        # Flecha derecha para avanzar
                        with cols[1]:
                            if st.session_state.index < len(data) - 1:
                                if st.button("→"):
                                    st.session_state.index += 1
                    else:
                        st.write("No hay imágenes disponibles para este índice.")
                elif 'Images_URL' not in data.columns:
                    st.write("Este archivo no contiene información de imágenes.")
                else:
                    st.write("Las imágenes para este índice no están en un formato válido.")
        
        else:
            column = st.selectbox("Seleccione la columna", data.columns)
            st.session_state.search_value = st.text_input(f"Ingrese el valor para buscar en la columna {column}", value=st.session_state.search_value)
            if st.button("Buscar"):
                query = st.session_state.search_value
                st.write(data[data[column].astype(str).str.contains(query, case=False, na=False)])


    else:
        st.error("No se pudo cargar el archivo. Formato no soportado.")
