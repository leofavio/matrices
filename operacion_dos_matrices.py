import streamlit as st
import numpy as np
import sympy as sp
from sympy import Matrix, symbols

def operacion_dos_matrices():
    st.header(":blue[Operaciones con Dos Matrices]", divider="blue")

    # Primera sección: Configuración de las matrices
    col_config_a, col_config_b = st.columns(2)

    with col_config_a:
        # Configuración de la matriz A
        subcol_dim_a1, subcol_dim_a2 = st.columns([2, 1])  # proporción 2:1
        with subcol_dim_a1:
            st.subheader("Matriz A")
            fila_columna1, fila_columna2 = st.columns(2)
            with fila_columna1:
                filas_a = st.slider("Filas A", min_value=1, max_value=6, value=3, key="filas_a")
            with fila_columna2:
                columnas_a = st.slider("Columnas A", min_value=1, max_value=6, value=3, key="columnas_a")

        # Configuración de la matriz B
        subcol_dim_b1, subcol_dim_b2 = st.columns([2, 1])  # proporción 2:1
        with subcol_dim_b1:
            st.subheader("Matriz B")
            fila_columna1, fila_columna2 = st.columns(2)
            with fila_columna1:
                filas_b = st.slider("Filas B", min_value=1, max_value=6, value=3, key="filas_b")
            with fila_columna2:
                columnas_b = st.slider("Columnas B", min_value=1, max_value=6, value=3, key="columnas_b")

    with col_config_b:
        st.success("""
        **INSTRUCCIONES**
        1. Configura las dimensiones y el tipo de matriz (numérica o algebraica).
        2. Ingresa los valores de la matriz. Si es un número imaginario usa 'i' o 'j'.
        3. Elige la operación que deseas realizar.
        4. Observa el resultado en la sección de resultados.
        """)
        tipo_matriz = st.radio("Tipo de Matrices", ["Numérica", "Algebraica"], horizontal=True, key="tipo_dos")

    # Segunda sección: Ingreso de matrices
    col_matriz_a, col_matriz_b = st.columns(2, border=True)

    # --- Matriz A ---
    with col_matriz_a:
        st.subheader(":blue[Matriz A]")
        if tipo_matriz == "Numérica":
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("Limpiar A 🧹", key="limpiar_a"):
                    for i in range(filas_a):
                        for j in range(columnas_a):
                            st.session_state[f"A_dos_{i}_{j}_num"] = 0.0
                    st.rerun()
            with btn_col2:
                if st.button("Aleatorio A 🎲", key="aleatorio_a"):
                    for i in range(filas_a):
                        for j in range(columnas_a):
                            st.session_state[f"A_dos_{i}_{j}_num"] = np.round(np.random.uniform(-10,10),2)
                    st.rerun()
            matriz_a = crear_matriz_numerica_compacta(filas_a, columnas_a, "A_dos")
        else:
            matriz_a = crear_matriz_algebraica_compacta(filas_a, columnas_a, "A_dos")

        st.subheader("Visualización Matriz A")
        if tipo_matriz == "Numérica":
            st.dataframe(matriz_a, use_container_width=True)
        else:
            st.latex(sp.latex(matriz_a))

    # --- Matriz B ---
    with col_matriz_b:
        st.subheader(":blue[Matriz B]")
        if tipo_matriz == "Numérica":
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("Limpiar B 🧹", key="limpiar_b"):
                    for i in range(filas_b):
                        for j in range(columnas_b):
                            st.session_state[f"B_dos_{i}_{j}_num"] = 0.0
                    st.rerun()
            with btn_col2:
                if st.button("Aleatorio B 🎲", key="aleatorio_b"):
                    for i in range(filas_b):
                        for j in range(columnas_b):
                            st.session_state[f"B_dos_{i}_{j}_num"] = np.round(np.random.uniform(-10,10),2)
                    st.rerun()
            matriz_b = crear_matriz_numerica_compacta(filas_b, columnas_b, "B_dos")
        else:
            matriz_b = crear_matriz_algebraica_compacta(filas_b, columnas_b, "B_dos")

        st.subheader("Visualización Matriz B")
        if tipo_matriz == "Numérica":
            st.dataframe(matriz_b, use_container_width=True)
        else:
            st.latex(sp.latex(matriz_b))

    # Tercera sección: Botones y Resultados
    col_botones, col_resultado = st.columns(2, border=True)

    # --- Botones de operaciones ---
    with col_botones:
        fila1 = st.container()
        with fila1:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Suma A + B", key="suma_btn", width="stretch", type="primary"):
                    calcular_operacion_dos("suma", matriz_a, matriz_b, tipo_matriz)
            with c2:
                if st.button("Resta A - B", key="resta_btn", width="stretch", type="primary"):
                    calcular_operacion_dos("resta", matriz_a, matriz_b, tipo_matriz)
        fila2 = st.container()
        with fila2:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Producto A × B", key="prod_ab_btn", width="stretch", type="primary"):
                    calcular_operacion_dos("producto", matriz_a, matriz_b, tipo_matriz)
            with c2:
                if st.button("Producto B × A", key="prod_ba_btn", width="stretch", type="primary"):
                    calcular_operacion_dos("producto", matriz_b, matriz_a, tipo_matriz)

    # --- Resultados ---
    with col_resultado:
        st.subheader(":blue[Resultado]")
        if "resultado_dos" in st.session_state:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            op = st.session_state.get("tipo_operacion_dos", "")
            valor = st.session_state.get("resultado_dos", "")

            if op == "error":
                st.error(valor)
            elif op == "suma":
                st.subheader("Suma A + B")
                if tipo_matriz == "Numérica":
                    st.write(valor)
                else:
                    st.latex(f"A + B = {sp.latex(valor)}")
            elif op == "resta":
                st.subheader("Resta A - B")
                if tipo_matriz == "Numérica":
                    st.write(valor)
                else:
                    st.latex(f"A - B = {sp.latex(valor)}")
            elif op == "producto":
                st.subheader("Producto de Matrices")
                if tipo_matriz == "Numérica":
                    st.write(valor)
                else:
                    st.latex(f"A \\times B = {sp.latex(valor)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
    # Sección de Paso a Paso (ocupa todo el ancho)
    if "pasos_dos" in st.session_state and st.session_state["pasos_dos"]:
        st.subheader(":blue[Paso a Paso]")
        
        with st.expander("Ver detalles del cálculo", expanded=True):
            for paso in st.session_state["pasos_dos"]:
                if paso.startswith("$$") and paso.endswith("$$"):
                    st.latex(paso[2:-2])  # Remover los $$ para latex
                elif paso.startswith("###"):
                    st.markdown(paso)
                else:
                    st.code(paso if "\n" in paso else paso, language="python" if "```" in paso else None)

def crear_matriz_numerica_compacta(filas, columnas, prefix="A"):
    matriz = np.zeros((filas, columnas))
    
    for i in range(filas):
        cols = st.columns(columnas)
        for j in range(columnas):
            with cols[j]:
                key = f"{prefix}_{i}_{j}_num"
                
                # Inicializar el valor en session_state si no existe
                if key not in st.session_state:
                    st.session_state[key] = 0.0
                
                # Usar el valor de session_state como valor por defecto
                matriz[i, j] = st.number_input(
                    f"{prefix}[{i+1},{j+1}]", 
                    value=st.session_state[key],
                    key=key,
                    label_visibility="collapsed"
                )
    return matriz

def crear_matriz_algebraica_compacta(filas, columnas, prefix="A"):
    matriz = sp.Matrix.zeros(filas, columnas)
    
    for i in range(filas):
        cols = st.columns(columnas)
        for j in range(columnas):
            with cols[j]:
                default_val = f"a_{i+1}{j+1}" if (i != j) else f"a_{i+1}"
                expr_str = st.text_input(
                    f"{prefix}[{i+1},{j+1}]", 
                    value=default_val,
                    key=f"{prefix}_{i}_{j}_sym",
                    label_visibility="collapsed"
                )
                try:
                    matriz[i, j] = sp.sympify(expr_str)
                except:
                    matriz[i, j] = sp.sympify(default_val)
    return matriz

def calcular_operacion_dos(operacion, matriz_a, matriz_b, tipo_matriz):
    try:
        pasos = []  # Lista para almacenar los pasos
        
        if operacion == "suma":
            if matriz_a.shape != matriz_b.shape:
                st.session_state["resultado_dos"] = "❌ Error: Las matrices deben tener las mismas dimensiones para la suma."
                st.session_state["tipo_operacion_dos"] = "error"
                return
            
            pasos.append("### Paso 1: Matriz A")
            if tipo_matriz == "Numérica":
                pasos.append(f"```\n{matriz_a}\n```")
                pasos.append("### Paso 2: Matriz B")
                pasos.append(f"```\n{matriz_b}\n```")
                pasos.append("### Paso 3: Sumar elemento por elemento: A[i,j] + B[i,j]")
                resultado = matriz_a + matriz_b
                pasos.append("Resultado A + B:")
                pasos.append(f"```\n{resultado}\n```")
            else:
                pasos.append(f"$$A = {sp.latex(matriz_a)}$$")
                pasos.append(f"$$B = {sp.latex(matriz_b)}$$")
                pasos.append("### Paso 3: Sumar elemento por elemento")
                resultado = matriz_a + matriz_b
                pasos.append(f"$$A + B = {sp.latex(resultado)}$$")
        
        elif operacion == "resta":
            if matriz_a.shape != matriz_b.shape:
                st.session_state["resultado_dos"] = "❌ Error: Las matrices deben tener las mismas dimensiones para la resta."
                st.session_state["tipo_operacion_dos"] = "error"
                return
            
            pasos.append("### Paso 1: Matriz A")
            if tipo_matriz == "Numérica":
                pasos.append(f"```\n{matriz_a}\n```")
                pasos.append("### Paso 2: Matriz B")
                pasos.append(f"```\n{matriz_b}\n```")
                pasos.append("### Paso 3: Restar elemento por elemento: A[i,j] - B[i,j]")
                resultado = matriz_a - matriz_b
                pasos.append("Resultado A - B:")
                pasos.append(f"```\n{resultado}\n```")
            else:
                pasos.append(f"$$A = {sp.latex(matriz_a)}$$")
                pasos.append(f"$$B = {sp.latex(matriz_b)}$$")
                pasos.append("### Paso 3: Restar elemento por elemento")
                resultado = matriz_a - matriz_b
                pasos.append(f"$$A - B = {sp.latex(resultado)}$$")
        
        elif operacion == "producto":
            if matriz_a.shape[1] != matriz_b.shape[0]:
                st.session_state["resultado_dos"] = "❌ Error: Columnas de A deben coincidir con filas de B."
                st.session_state["tipo_operacion_dos"] = "error"
                return
            
            pasos.append("### Paso 1: Matriz A")
            if tipo_matriz == "Numérica":
                pasos.append(f"```\n{matriz_a}\n```")
                pasos.append("### Paso 2: Matriz B")
                pasos.append(f"```\n{matriz_b}\n```")
                pasos.append("### Paso 3: Multiplicación matricial")
                pasos.append("Cada elemento C[i,j] = Σ (A[i,k] × B[k,j]) para k = 1 to n")
                resultado = np.matmul(matriz_a, matriz_b)
                pasos.append("Resultado A × B:")
                pasos.append(f"```\n{resultado}\n```")
            else:
                pasos.append(f"$$A = {sp.latex(matriz_a)}$$")
                pasos.append(f"$$B = {sp.latex(matriz_b)}$$")
                pasos.append("### Paso 3: Multiplicación matricial")
                resultado = matriz_a * matriz_b
                pasos.append(f"$$A \\times B = {sp.latex(resultado)}$$")

        st.session_state["resultado_dos"] = resultado
        st.session_state["tipo_operacion_dos"] = operacion
        st.session_state["pasos_dos"] = pasos

    except Exception as e:
        st.session_state["resultado_dos"] = f"❌ Error al calcular la operación: {e}"
        st.session_state["tipo_operacion_dos"] = "error"