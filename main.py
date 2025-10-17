import streamlit as st
from introduccion import mostrar_introduccion
from operacion_una_matriz import operacion_una_matriz
from operacion_dos_matrices import operacion_dos_matrices
from sistema_ecuaciones import mostrar_sistema_ecuaciones

# Configuración de la página
st.set_page_config(
    page_title="Calculadora Matemática",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Sidebar con páginas
    with st.sidebar:
        st.image("loguito_a.png", width=200)
        st.title("🧮 Calculadora Matemática")
        page = st.radio(
            "Selecciona una página:", 
            [
                "Operación con Matriz", 
                "Operaciones con Matrices",
                "Sistema de Ecuaciones"
            ]
        )
    
    # Navegación entre páginas
    if page == "Operación con Matriz":
        operacion_una_matriz()
    elif page == "Operaciones con Matrices":
        operacion_dos_matrices()
    elif page == "Sistema de Ecuaciones":
        mostrar_sistema_ecuaciones()

if __name__ == "__main__":
    main()
