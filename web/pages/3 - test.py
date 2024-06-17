import streamlit as st
import pandas as pd
import os
import ast

## Funciones
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
        
# Función para guardar el resultado de un botón
def click_button(name):
    st.session_state[name] = True

## Aplicación

st.title("Buscador")

# Obtener lista de archivos en la carpeta de datos
cwd = os.getcwd()
data_folder = os.path.join(cwd, 'web', 'pages', 'data')
files = os.listdir(data_folder)
files = [file for file in files if file.endswith(('csv', 'xlsx'))]

# Selección de archivo
st.header("Seleccione el archivo")
selected_file = st.selectbox("Archivos disponibles", files, key="selected_file")

# Inicializamos los botones
if 'Buscar' not in st.session_state:
    st.session_state.Buscar = False

if 'Buscar2' not in st.session_state:
    st.session_state.Buscar2 = False

if '->' not in st.session_state:
    st.session_state.-> = False

if '<-' not in st.session_state:
    st.session_state.<- = False

# Inicializamos indice de imagen
if "act" not in st.session_state:
            st.session_state.act = 0

# Cuando se seleccione un archivo
if st.session_state.selected_file:

    # Cargamos los datos
    file_path = os.path.join(data_folder, selected_file)
    data = load_file(file_path)

    # Selección de si buscamos por índice o por columna
    search_option = st.radio("Buscar por", ('Índice', 'Nombre de columna'), key="search_option")

    # Busqueda por índice
    if st.session_state.search_option == 'Índice':
        
        index = st.number_input("Ingrese el índice", min_value=0, max_value=len(data)-1, key="file_index")

        st.button('Buscar', on_click=click_button("Buscar"), type="primary")

        # Cuando se dé al botón de buscar
        if st.session_state.Buscar:

            # Se muestran los datos del índice
            st.write(data.iloc[st.session_state.file_index])

            # Si entre las columnas, el dataframe tiene imágenes
            if 'Images_URL' in data.columns:

                # Reseteamos el índice de la imagen a 0
                st.session_state.act = 0

                # Mostramos las imágenes
                st.markdown("**Imágenes:**")

                # Lista de todas las imágenes del índice
                img_list = ast.literal_eval(data.loc[st.session_state.file_index, 'Images_URL'])

                # Mostrar la imagen actual
                st.image(img_list[st.session_state["act"]].strip(), caption=f"Imagen {st.session_state['act'] + 1} de {len(img_list)}")

                # Añadir flechas para navegar entre las imágenes
                cols = st.columns(2)  # 2 columnas para las flechas

                # Flecha izquierda para retroceder
                 with cols[0]:
                    st.button("←", on_click=click_button("<-"))
                    if st.session_state.<-:
                        st.session_state["act"] = (st.session_state["act"] - 1) 
                        if st.session_state["act"] < 0:
                            st.session_state["act"] = len(img_list) - 1
                
                # Flecha derecha para avanzar
                with cols[1]: 
                    st.button("→", on_click=click_button("->"))
                    if st.session_state.->:
                        st.session_state["act"] = (st.session_state["act"] + 1)
                        if st.session_state["act"] > len(img_list) - 1:
                            st.session_state["act"] = 0
                        

    # Si se busca por el nombre de la columna
    else:
        column = st.selectbox("Seleccione la columna", data.columns, key="column")
        query = st.text_input(f"Ingrese el valor para buscar en la columna {st.session_state.column}")

        st.button('Buscar', on_click=click_button("Buscar2"), type="primary")
        
        if st.session_state.Buscar2:
            st.write(data[data[st.session_state.column].astype(str).str.contains(query, case=False, na=False)])
