import streamlit as st
import pandas as pd
import os
import ast

## Init botón
if "buscar" not in st.session_state:
    st.session_state.buscar = False

## Funciones
# Función para cargar archivos
def load_file(file_path):
    return pd.read_csv(file_path)

def callback():
    st.session_state.buscar=True

## Aplicación
st.title("Buscador")

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
        
        index = st.number_input("Ingrese el índice", min_value=0, max_value=len(data)-1, step=1)
        
        if st.button("Buscar", on_click=callback()) or st.session_state.buscar:
            st.session_state.image_index = 0  # Reiniciar el índice de imagen al buscar un nuevo índice
            st.session_state.selected_index = index  # Guardar el índice seleccionado
            
            st.write(data.iloc[index])
            
            if 'Images_URL' in data.columns:
                img_list = ast.literal_eval(data.loc[index, 'Images_URL'])
                current_image_index = st.session_state.image_index
                
                # Mostrar la imagen actual
                st.image(img_list[current_image_index].strip(), caption="{} de {}".format(current_image_index + 1, len(img_list)))
                
                # Añadir flechas para navegar entre las imágenes
                cols = st.columns(2)  # 2 columnas para las flechas
                
                # Flecha izquierda para retroceder
                with cols[0]:
                    if st.button("←"):
                        if st.session_state.image_index > 0:
                            st.session_state.image_index -= 1
                        else:
                            st.session_state.image_index = len(img_list) - 1
                        st.rerun()
                
                # Flecha derecha para avanzar
                with cols[1]:
                    if st.button("→"):
                        if st.session_state.image_index < len(img_list) - 1:
                            st.session_state.image_index += 1
                        else:
                            st.session_state.image_index = 0
                        st.rerun()

    # Nombre de columna
    else:
        column = st.selectbox("Seleccione la columna", data.columns)
        query = st.text_input(f"Ingrese el valor para buscar en la columna {column}")
        if st.button("Buscar"):
            st.write(data[data[column].astype(str).str.contains(query, case=False, na=False)])

