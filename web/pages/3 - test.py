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

st.title(f"Buscador de imágenes")

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
        st.write(f"Cargado archivo: {selected_file}")
        
        # Buscador por índice
        index = st.number_input("Ingrese el índice", min_value=0, max_value=len(data)-1, step=1)
        if st.button("Buscar"):
            st.write(data.iloc[index])
            
            # Mostrar las imágenes si están disponibles
            if 'Images_URL' in data.columns and isinstance(data.loc[index, 'Images_URL'], str):
                st.markdown("**Imágenes:**")
                img_list = ast.literal_eval(data.loc[index, 'Images_URL'])
                if img_list:  # Verificar si hay imágenes en la lista
                    st.image(img_list[0].strip(), caption="1 de {}".format(len(img_list)))
                    
                    # Añadir flechas para navegar entre las imágenes
                    cols = st.columns(2)  # 2 columnas para las flechas
                    
                    if st.button("←"):
                        st.session_state.img_index -= 1
                        if st.session_state.img_index < 0:
                            st.session_state.img_index = len(img_list) - 1
                    
                    if st.button("→"):
                        st.session_state.img_index += 1
                        if st.session_state.img_index >= len(img_list):
                            st.session_state.img_index = 0
                    
                    # Mostrar la imagen actual
                    st.image(img_list[st.session_state.img_index].strip(), caption=f"{st.session_state.img_index + 1} de {len(img_list)}")
                
                else:
                    st.write("No hay imágenes disponibles para este índice.")
            else:
                st.write("Este archivo no contiene información de imágenes o el formato no es válido.")
