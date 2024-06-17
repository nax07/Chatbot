import streamlit as st
import pandas as pd
import os
import ast

## Init


## Funciones
# Función para cargar archivos
def load_file(file_path):
    return pd.read_csv(file_path)


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

        result = 0
      
        st.write(result)
          
        result = st.button("Press me")

        st.write(result)
        
        
    # Nombre de columna
    else:
        column = st.selectbox("Seleccione la columna", data.columns)
        query = st.text_input(f"Ingrese el valor para buscar en la columna {column}")
        if st.button("Buscar"):
            st.write(data[data[column].astype(str).str.contains(query, case=False, na=False)])



