import streamlit as st

def mostrar_introduccion():
    st.title("Bienvenido a la Calculadora Matemática")

    # --- Primer bloque: Qué hace la calculadora ---
    st.subheader("🔹 Funciones principales")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Operaciones con matrices**
          - Determinante, inversa, transpuesta
          - Traza, rango, multiplicación por escalar
        - **Operaciones con dos matrices**
          - Suma, resta, multiplicación, producto
        """)
    with col2:
        st.markdown("""
        - **Sistemas de ecuaciones**
          - Resolver sistemas lineales paso a paso
        """)

    # --- Segundo bloque: Tipos de matrices ---
    st.subheader("🔹 Tipos de matrices")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("1. **Numéricas**: valores reales")
    with col2:
        st.markdown("2. **Algebraicas**: expresiones simbólicas")

    # --- Tercer bloque: Instrucciones ---
    st.subheader("🔹 Cómo usar")
    cols = st.columns(3)
    instrucciones = [
        "Selecciona la página de la operación",
        "Configura dimensiones y tipo de matriz",
        "Ingresa los valores",
        "Elige la operación",
        "Observa el resultado"
    ]
    for i, col in enumerate(cols * 2):  # duplicamos para acomodar 5 instrucciones
        if i < len(instrucciones):
            col.markdown(f"{i+1}. {instrucciones[i]}")

    # --- Cuarto bloque: Ejemplos ---
    st.subheader("🔹 Ejemplos de uso")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - Determinante de matriz 3x3
        - Inversa de matriz cuadrada
        - Suma/multiplicación de matrices
        """)
    with col2:
        st.markdown("""
        - Resolver sistemas lineales
        - Trabajar con matrices simbólicas
        """)

    st.info("💡 Consejo: Usa la navegación del sidebar para acceder a cada operación.")
