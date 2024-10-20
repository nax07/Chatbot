import streamlit as st

st.title('Chatbot con documentación propia usando LLM')
st.subheader('Por Ignacio Domínguez Espinosa')

st.subheader("Introducción")

st.write("""
Este chatbot es un asistente virtual diseñado para responder consultas sobre ejercicio y deporte de manera rápida y eficiente.
La aplicación permite elegir entre 3 modelos diferentes de lenguaje natural, 4 modos de procesamiento de peticiones y es compatible con una gran variedad de idiomas.
""")

st.subheader("¿Cómo utilizar la aplicación?")
st.write("""
Para ejecutar correctamente la aplicación, primero se deben configurar las opciones del chatbot. Estas se encuentran en el menú lateral de la izquierda, en la sección de "Opciones".
Hay tres parámetros que se pueden modificar:
""")

st.markdown("""
1. **Idioma**: Define el idioma en el que se envía la consulta al chatbot y el idioma en el que éste responderá. Los idiomas disponibles son: Español, Inglés, Francés, Alemán, Italiano, Ruso, Chino (Mandarín), Árabe e Hindi.
2. **Modelo**: Esta opción determina qué modelo de lenguaje natural se usará para procesar la petición. Los modelos disponibles son: Gpt2-xl, Cohere y Llama3. Para los dos últimos se necesita una clave "API".

    - **Cohere**: Para obtener la clave, hay que acceder a la página oficial de Cohere (https://cohere.com/), iniciar sesión con el correo electrónico, acceder al dashboard y en la sección "API keys", crear una "New Trial Key". Estas claves son gratuitas pero limitadas.

    - **Llama3**: Para usar este modelo, se utiliza la plataforma Together.ai (https://www.together.ai/), que nos permite tomar el modelo de forma online sin necesidad de descargarlo. Se debe iniciar sesión con el correo electrónico, acceder a la sección de dashboard y generar una "API key". Estas claves tienen un límite de uso de $5.
    
3. **Tipo de procesamiento**: Uno de los puntos fuertes de la aplicación es poder utilizar diferentes tipos de procesamiento para una misma pregunta. Las opciones disponibles son:

    1. **Regular processing**: La petición del usuario se envía directamente al modelo de lenguaje natural.
    2. **Advanced prompts processing**: Utiliza la librería Langchain para estructurar la petición antes de enviarla al modelo.
    3. **Regular RAG (Retrieval-Augmented Generation)**: Se añade a la petición de Langchain un contexto. Este se calcula utilizando documentos de deporte relacionados con la petición.
    4. **Multi-Query RAG**: Mejora la búsqueda de documentos de la opción anterior al reinterpretar la petición del usuario para obtener mejores resultados.

""")
st.markdown("""
En las dos últimas opciones, los documentos utilizados para añadir contexto provienen de:



Una vez se han cargado las configuraciones, haz clic en el botón "Confirmar configuraciones" para que la aplicación cargue la información correcta. Después de la carga, puedes comenzar a interactuar con el chatbot a través del cuadro de texto ubicado en la parte inferior de la página.
""")
