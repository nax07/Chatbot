import streamlit as st
import pandas as pd
import os
import ast

## Init
# Página
if "pag" not in st.session_state:
    st.session_state.pag = 0

# Botón
if "buscar" not in st.session_state:
    st.session_state.buscar = False

## Funciones
# Función para cargar archivos
def load_file(file_path):
    return pd.read_csv(file_path)

def callback():
    st.session_state.buscar = True

# Funciones para editar el session_state


## Aplicación
st.title("Buscador")

st.write(st.session_state)

# Obtener lista de archivos en la carpeta de datos
cwd = os.getcwd()
data_folder = os.path.join(cwd, 'web', 'pages', 'data')
files = os.listdir(data_folder)
files = [file for file in files if file.endswith(('csv', 'xlsx'))]

# Seleccionar archivo
st.header("Seleccione el archivo")
selected_file = st.selectbox("Archivos disponibles", files)

if selected_file:

    # Carga de datos
    file_path = os.path.join(data_folder, selected_file)
    data = load_file(file_path)
    
    # Buscador por índice o nombre de columna
    search_option = st.radio("Buscar por", ('Índice', 'Nombre de columna'))

    # Índice
    if search_option == 'Índice':
        
        index = st.number_input("Ingrese el índice", min_value=0, max_value=len(data)-1)

        result = st.button("Buscar")
        if result:
            st.session_state.buscar = True
            st.session_state.clicked = True
            st.session_state.pag = 0

        if st.session_state.buscar or st.session_state.clicked:
            if st.session_state.clicked:
                st.session_state.pag = 0
            
            st.write(data.iloc[index])
            
            if 'Images_URL' in data.columns:
                # Lista de imágenes
                img_list = ast.literal_eval(data.loc[index, 'Images_URL'])

                # Mostrar la imagen actual
                st.image(img_list[st.session_state.pag].strip(), caption="{} de {}".format(st.session_state.pag + 1, len(img_list)))
                
                # Añadir flechas para navegar entre las imágenes
                cols = st.columns(3)  # 2 columnas para las flechas
                
                # Flecha izquierda para retroceder
                with cols[1]:
                    if st.button("←", on_click=callback()):
                        if st.session_state.pag > 0:
                            st.session_state.pag -= 1
                        else:
                            st.session_state.pag = len(img_list) - 1
                
                # Flecha derecha para avanzar
                with cols[2]:
                    if st.button("→", on_click=callback()):
                        if st.session_state.pag < len(img_list) - 1:
                            st.session_state.pag += 1
                        else:
                            st.session_state.pag = 0
        
      

    # Nombre de columna
    else:
        column = st.selectbox("Seleccione la columna", data.columns)
        query = st.text_input(f"Ingrese el valor para buscar en la columna {column}")
        if st.button("Buscar"):
            st.write(data[data[column].astype(str).str.contains(query, case=False, na=False)])

st.session_state.clicked = False
