import streamlit as st
import numpy as np
import sympy as sp
from sympy import Matrix, symbols

def operacion_una_matriz():
    st.header(":blue[Operaciones con una Matriz]", divider="blue")

    # --- Funciones de callback para cambios de dimensión ---
    def on_dim_change():
        # Reiniciar solo la sección de pasos y métodos (sin tocar valores de la matriz)
        for key in ["pasos", "metodo_determinante", "metodo_inversa", "inversa_existe", "tipo_operacion", "resultado"]:
            if key in st.session_state:
                del st.session_state[key]

    # Primera sección: Configuración de la matriz
    col_izq, col_der = st.columns(2)

    # Columna izquierda: Dimensiones y Tipo
    with col_izq:
        subcol_dim, subcol_tipo = st.columns([2, 1])  # proporción 2:1

        # Dimensiones (Filas y Columnas en la misma fila)
        with subcol_dim:
            st.subheader("Dimensión de la Matriz")
            fila_columna1, fila_columna2 = st.columns(2)

            with fila_columna1:
                # Agrego on_change para reiniciar pasos al cambiar dimensiones
                filas = st.slider("Filas", min_value=1, max_value=6, value=3, key="filas_una", on_change=on_dim_change)
            with fila_columna2:
                columnas = st.slider("Columnas", min_value=1, max_value=6, value=3, key="columnas_una", on_change=on_dim_change)

        # Tipo de matriz
        with subcol_tipo:
            st.subheader("Tipo de Matriz")
            tipo_matriz = st.radio(
                "Selecciona el tipo:",
                ["Numérica", "Algebraica"],
                horizontal=True,
                key="tipo_una"
            )

    # Columna derecha: instrucciones en bloque success
    with col_der:
        st.success("""
        **INSTRUCCIONES**
        1. Configura las dimensiones y el tipo de matriz (numérica o algebraica).
        2. Ingresa los valores de la matriz. Si es un número imaginario usa 'i' o 'j'.
        3. Elige la operación que deseas realizar.
        4. Observa el resultado en la sección de resultados.
        """)

    # Segunda sección: Ingreso de la matriz
    col_matriz, col_visualizacion = st.columns(2, border=True)

    with col_matriz:
        st.subheader(":blue[Ingreso de Datos de la Matriz]")

        # Solo mostrar los botones si la matriz es numérica
        if tipo_matriz == "Numérica":
            btn_col1, btn_col2 = st.columns(2)

            with btn_col1:
                if st.button("Limpiar Resultados 🧹",width="stretch"):
                    for i in range(filas):
                        for j in range(columnas):
                            key = f"A_{i}_{j}_num"
                            st.session_state[key] = 0
                    st.rerun()

            with btn_col2:
                if st.button("Generar Aleatorios 🎲",width="stretch"):
                    for i in range(filas):
                        for j in range(columnas):
                            key = f"A_{i}_{j}_num"
                            st.session_state[key] = np.random.randint(-10, 11)
                    st.rerun()

        # Crear la matriz usando los valores actuales de session_state
        if tipo_matriz == "Numérica":
            matriz_a = crear_matriz_numerica_compacta(filas, columnas, "A")
        else:
            matriz_a = crear_matriz_algebraica_compacta(filas, columnas, "A")

    # Mostrar la matriz
    with col_visualizacion:
        st.subheader(":blue[Matriz Ingresada]")
        if tipo_matriz == "Numérica":
            st.dataframe(matriz_a, hide_index=False)
        else:
            st.latex(sp.latex(matriz_a))

    # Tercera sección: Botones de operaciones y resultados
    col_botones, col_resultado = st.columns(2,border=True)  # izquierda botones, derecha resultados

    # Columna izquierda: botones organizados en dos filas
    with col_botones:
        # Inicializar estado del escalar si no existe
        if "mostrar_escalar" not in st.session_state:
            st.session_state["mostrar_escalar"] = False
        if "escalar_valor" not in st.session_state:
            st.session_state["escalar_valor"] = 2  # valor por defecto entero

        # Fila 1
        fila1 = st.container()
        with fila1:
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Determinante", key="det_btn",width="stretch", type="primary"):
                    st.session_state["mostrar_escalar"] = False
                    # Reiniciar pasos cuando se elige nueva operación
                    if "pasos" in st.session_state:
                        del st.session_state["pasos"]
                    if "metodo_determinante" in st.session_state:
                        del st.session_state["metodo_determinante"]
                    calcular_operacion("determinante", matriz_a, tipo_matriz)
            with c2:
                if st.button("Inversa", key="inv_btn",width="stretch", type="primary"):
                    st.session_state["mostrar_escalar"] = False
                    # Reiniciar pasos cuando se elige nueva operación
                    if "pasos" in st.session_state:
                        del st.session_state["pasos"]
                    if "metodo_inversa" in st.session_state:
                        del st.session_state["metodo_inversa"]
                    calcular_operacion("inversa", matriz_a, tipo_matriz)
            with c3:
                if st.button("Transpuesta", key="trans_btn",width="stretch", type="primary"):
                    st.session_state["mostrar_escalar"] = False
                    # Reiniciar pasos cuando se elige nueva operación
                    if "pasos" in st.session_state:
                        del st.session_state["pasos"]
                    calcular_operacion("transpuesta", matriz_a, tipo_matriz)

        # Fila 2
        fila2 = st.container()
        with fila2:
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Traza", key="traza_btn",width="stretch", type="primary"):
                    st.session_state["mostrar_escalar"] = False
                    # Reiniciar pasos cuando se elige nueva operación
                    if "pasos" in st.session_state:
                        del st.session_state["pasos"]
                    calcular_operacion("traza", matriz_a, tipo_matriz)
            with c2:
                if st.button("Rango", key="rango_btn",width="stretch", type="primary"):
                    st.session_state["mostrar_escalar"] = False
                    # Reiniciar pasos cuando se elige nueva operación
                    if "pasos" in st.session_state:
                        del st.session_state["pasos"]
                    calcular_operacion("rango", matriz_a, tipo_matriz)
            with c3:
                if st.button("Multiplicación Escalar", key="escalar_btn",width="stretch", type="primary"):
                    st.session_state["mostrar_escalar"] = True
                    # Reiniciar pasos cuando se elige nueva operación
                    if "pasos" in st.session_state:
                        del st.session_state["pasos"]

        # Input para escalar (solo si se seleccionó)
        if st.session_state.get("mostrar_escalar", False):
            if tipo_matriz == "Numérica":
                escalar_valor = st.number_input(
                    "Valor del escalar:", 
                    value=st.session_state.get("escalar_valor", 2),
                    key="escalar_valor_input",
                    step=1  # Para que sea entero por defecto
                )
                st.session_state["escalar_valor"] = escalar_valor
                if st.button("Calcular Multiplicación", key="calc_escalar",type="primary"):
                    calcular_operacion("escalar", matriz_a, tipo_matriz, escalar_valor)
            else:
                escalar_expr = st.text_input(
                    "Expresión del escalar:", 
                    value="k", 
                    key="escalar_expr_input"
                )
                if st.button("Calcular Multiplicación", key="calc_escalar_sym"):
                    try:
                        escalar_sym = sp.sympify(escalar_expr)
                        calcular_operacion("escalar", matriz_a, tipo_matriz, escalar_sym)
                    except:
                        st.error("Expresión no válida para el escalar")

    # Columna derecha: resultados
    with col_resultado:
        st.subheader(":blue[Resultado]")
        if "resultado" in st.session_state:
            
            if st.session_state["tipo_operacion"] == "determinante":
                if tipo_matriz == "Numérica":
                    st.success(f"El determinante de la matriz es: {st.session_state['resultado']}")
                    
                    # Mostrar opciones de método para determinante usando selectbox
                    opciones_metodos = []
                    if matriz_a.shape[0] == matriz_a.shape[1] == 3:
                        opciones_metodos.append("Sarrus")
                    opciones_metodos.extend(["Expansión por cofactores", "Reducción por filas"])

                    # Inicializar valor por defecto si no existe
                    if "metodo_determinante" not in st.session_state:
                        if "Sarrus" in opciones_metodos:
                            st.session_state["metodo_determinante"] = "Sarrus"
                        else:
                            st.session_state["metodo_determinante"] = "Expansión por cofactores"

                    metodo_actual = st.session_state.get("metodo_determinante", opciones_metodos[0])
                    
                    metodo = st.selectbox(
                        "Método de cálculo:",
                        opciones_metodos,
                        index=opciones_metodos.index(metodo_actual) if metodo_actual in opciones_metodos else 0,
                        key="metodo_det_select"
                    )
                    
                    # Si cambió el método, recalcular
                    if metodo != st.session_state.get("metodo_determinante"):
                        st.session_state["metodo_determinante"] = metodo
                        calcular_operacion("determinante", matriz_a, tipo_matriz)
                        st.rerun()
                    
                else:
                    st.latex(f"\\det(A) = {sp.latex(st.session_state['resultado'])}")

            elif st.session_state["tipo_operacion"] == "inversa":
                if tipo_matriz == "Numérica":
                    if st.session_state.get("inversa_existe", True):
                        st.write(st.session_state["resultado"])
                        
                        # Mostrar opciones de método para inversa usando selectbox
                        opciones_metodos = ["Adjunta/cofactores", "Gauss-Jordan"]
                        
                        if "metodo_inversa" not in st.session_state:
                            st.session_state["metodo_inversa"] = "Adjunta/cofactores"
                        
                        metodo_actual = st.session_state.get("metodo_inversa", "Adjunta/cofactores")
                        
                        metodo = st.selectbox(
                            "Método de cálculo:",
                            opciones_metodos,
                            index=opciones_metodos.index(metodo_actual),
                            key="metodo_inv_select"
                        )
                        
                        # Si cambió el método, recalcular
                        if metodo != st.session_state.get("metodo_inversa"):
                            st.session_state["metodo_inversa"] = metodo
                            calcular_operacion("inversa", matriz_a, tipo_matriz)
                            st.rerun()
                    else:
                        st.error("La matriz no es invertible")
                else:
                    st.latex(f"A^{{-1}} = {sp.latex(st.session_state['resultado'])}")

            elif st.session_state["tipo_operacion"] == "transpuesta":
                if tipo_matriz == "Numérica":
                    st.write(st.session_state["resultado"])
                else:
                    st.latex(f"A^T = {sp.latex(st.session_state['resultado'])}")

            elif st.session_state["tipo_operacion"] == "traza":
                if tipo_matriz == "Numérica":
                    st.success(f"La traza de la matriz es: {st.session_state['resultado']}")
                else:
                    st.latex(f"\\operatorname{{tr}}(A) = {sp.latex(st.session_state['resultado'])}")

            elif st.session_state["tipo_operacion"] == "rango":
                st.success(f"El rango de la matriz es: {st.session_state['resultado']}")

            elif st.session_state["tipo_operacion"] == "escalar":
                if tipo_matriz == "Numérica":
                    st.write(st.session_state["resultado"])
                else:
                    st.latex(f"{sp.latex(st.session_state['escalar_valor'])} \\times A = {sp.latex(st.session_state['resultado'])}")

    # ✅ NUEVA SECCIÓN: PASO A PASO (después de mostrar el resultado principal)
    if "pasos" in st.session_state and st.session_state["pasos"]:
        st.subheader("🧮 Paso a Paso del Cálculo")
        
        with st.expander("**Ver detalles del procedimiento**", expanded=True):
            for paso in st.session_state["pasos"]:
                if paso.startswith("$$") and paso.endswith("$$"):
                    st.latex(paso[2:-2])  # Para fórmulas LaTeX
                elif paso.startswith("###"):
                    st.markdown(paso)  # Para títulos
                elif "```" in paso:
                    # Para bloques de código/matrices
                    contenido = paso.replace("```", "").strip()
                    st.code(contenido, language="python")
                else:
                    st.write(paso)  # Para texto normal

def crear_matriz_numerica_compacta(filas, columnas, prefix="A"):
    matriz = np.zeros((filas, columnas), dtype=int)
    
    for i in range(filas):
        cols = st.columns(columnas)
        for j in range(columnas):
            with cols[j]:
                key = f"{prefix}_{i}_{j}_num"
                
                # Inicializar el valor en session_state si no existe
                if key not in st.session_state:
                    st.session_state[key] = 0
                
                # Usar el valor de session_state como valor por defecto
                matriz[i, j] = st.number_input(
                    f"{prefix}[{i+1},{j+1}]", 
                    value=st.session_state[key],
                    key=key,
                    label_visibility="collapsed",
                    step=1  # Para que sea entero por defecto
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

def calcular_operacion(operacion, matriz, tipo_matriz, escalar_valor=None):
    try:
        # Reiniciar pasos cada vez que se calcula una nueva operación
        pasos = []
        
        if operacion == "determinante":
            if matriz.shape[0] != matriz.shape[1]:
                st.error("El determinante solo está definido para matrices cuadradas.")
                return
            
            if tipo_matriz == "Numérica":
                pasos.append("### Paso 1: Matriz original")
                pasos.append(f"```\n{matriz}\n```")
                
                # Usar el método seleccionado
                metodo = st.session_state.get("metodo_determinante", "Expansión por cofactores")
                metodo_norm = metodo.lower()

                if "sarr" in metodo_norm:
                    # Sarrus (expuesto paso a paso)
                    resultado, pasos_det = calcular_determinante_sarrus(matriz)
                    pasos.extend(pasos_det)
                elif "expans" in metodo_norm or "cofactor" in metodo_norm:
                    resultado, pasos_det = calcular_determinante_expansion(matriz)
                    pasos.extend(pasos_det)
                else:
                    # Reducción por filas
                    resultado, pasos_det = calcular_determinante_reduccion(matriz)
                    pasos.extend(pasos_det)
                    
            else:
                pasos.append("### Paso 1: Matriz original")
                pasos.append(f"$${sp.latex(matriz)}$$")
                resultado = matriz.det()
                pasos.append("### Paso 2: Cálculo del determinante simbólico")
                pasos.append(f"$$\\det(A) = {sp.latex(resultado)}$$")
            
            st.session_state["resultado"] = resultado
            st.session_state["tipo_operacion"] = "determinante"
            st.session_state["pasos"] = pasos
        
        elif operacion == "inversa":
            if matriz.shape[0] != matriz.shape[1]:
                st.error("La matriz inversa solo está definida para matrices cuadradas.")
                return
            
            if tipo_matriz == "Numérica":
                pasos.append("### Paso 1: Matriz original")
                pasos.append(f"```\n{matriz}\n```")
                
                # Verificar si es invertible
                det = np.linalg.det(matriz)
                pasos.append("### Paso 2: Verificar invertibilidad")
                pasos.append(f"det(A) = {det}")
                
                if abs(det) < 1e-10:
                    st.session_state["inversa_existe"] = False
                    st.error("La matriz no es invertible (determinante ≈ 0)")
                    return
                
                st.session_state["inversa_existe"] = True
                
                # Usar el método seleccionado
                metodo = st.session_state.get("metodo_inversa", "Adjunta/cofactores")
                metodo_norm = metodo.lower()
                
                if "adj" in metodo_norm or "cofactor" in metodo_norm:
                    resultado, pasos_inv = calcular_inversa_adjunta(matriz)
                    pasos.extend(pasos_inv)
                else:
                    resultado, pasos_inv = calcular_inversa_gauss_jordan(matriz)
                    pasos.extend(pasos_inv)
                    
            else:
                pasos.append("### Paso 1: Matriz original")
                pasos.append(f"$${sp.latex(matriz)}$$")
                
                pasos.append("### Paso 2: Calcular matriz inversa simbólica")
                resultado = matriz.inv()
                pasos.append(f"$$A^{{-1}} = {sp.latex(resultado)}$$")
            
            st.session_state["resultado"] = resultado
            st.session_state["tipo_operacion"] = "inversa"
            st.session_state["pasos"] = pasos
        
        elif operacion == "transpuesta":
            pasos.append("### Paso 1: Matriz original")
            if tipo_matriz == "Numérica":
                pasos.append(f"```\n{matriz}\n```")
                resultado = matriz.T
                pasos.append("### Paso 2: Intercambiar filas por columnas")
                pasos.append("Matriz transpuesta:")
                pasos.append(f"```\n{resultado}\n```")
            else:
                pasos.append(f"$${sp.latex(matriz)}$$")
                resultado = matriz.T
                pasos.append("### Paso 2: Intercambiar filas por columnas")
                pasos.append(f"$$A^T = {sp.latex(resultado)}$$")
            
            st.session_state["resultado"] = resultado
            st.session_state["tipo_operacion"] = "transpuesta"
            st.session_state["pasos"] = pasos
        
        elif operacion == "traza":
            if matriz.shape[0] != matriz.shape[1]:
                st.error("La traza solo está definida para matrices cuadradas.")
                return
            
            pasos.append("### Paso 1: Matriz original")
            if tipo_matriz == "Numérica":
                pasos.append(f"```\n{matriz}\n```")
                resultado = np.trace(matriz)
                pasos.append("### Paso 2: Sumar elementos de la diagonal principal")
                diagonal = [matriz[i,i] for i in range(matriz.shape[0])]
                pasos.append(f"Diagonal: {diagonal}")
                pasos.append(f"Traza = {' + '.join(map(str, diagonal))} = {resultado}")
            else:
                pasos.append(f"$${sp.latex(matriz)}$$")
                resultado = matriz.trace()
                pasos.append("### Paso 2: Sumar elementos de la diagonal principal")
                pasos.append(f"$$\\operatorname{{tr}}(A) = {sp.latex(resultado)}$$")
            
            st.session_state["resultado"] = resultado
            st.session_state["tipo_operacion"] = "traza"
            st.session_state["pasos"] = pasos
        
        elif operacion == "rango":
            pasos.append("### Paso 1: Matriz original")
            if tipo_matriz == "Numérica":
                pasos.append(f"```\n{matriz}\n```")
                resultado, pasos_rango = calcular_rango_pasos(matriz)
                pasos.extend(pasos_rango)
            else:
                pasos.append(f"$${sp.latex(matriz)}$$")
                resultado = matriz.rank()
                pasos.append("### Paso 2: Calcular rango simbólico")
                pasos.append(f"Rango = {resultado}")
            
            st.session_state["resultado"] = resultado
            st.session_state["tipo_operacion"] = "rango"
            st.session_state["pasos"] = pasos
        
        elif operacion == "escalar":
            pasos.append("### Paso 1: Matriz original")
            if tipo_matriz == "Numérica":
                pasos.append(f"```\n{matriz}\n```")
                pasos.append(f"### Paso 2: Multiplicar por escalar {escalar_valor}")
                resultado = escalar_valor * matriz
                pasos.append("Matriz resultante:")
                pasos.append(f"```\n{resultado}\n```")
            else:
                pasos.append(f"$${sp.latex(matriz)}$$")
                pasos.append(f"### Paso 2: Multiplicar por escalar {sp.latex(escalar_valor)}")
                resultado = escalar_valor * matriz
                pasos.append(f"$${sp.latex(escalar_valor)} \\times A = {sp.latex(resultado)}$$")
            
            st.session_state["resultado"] = resultado
            st.session_state["tipo_operacion"] = "escalar"
            st.session_state["escalar_valor"] = escalar_valor
            st.session_state["pasos"] = pasos
    
    except Exception as e:
        st.error(f"Error al calcular la operación: {e}")

# Funciones auxiliares para cálculos paso a paso
def calcular_determinante_sarrus(matriz):
    pasos = []
    # Suponer 3x3
    n = matriz.shape[0]
    if n != 3:
        pasos.append("Sarrus solo es aplicable para matrices 3x3; se usará otro método.")
        return calcular_determinante_expansion(matriz)  # delegar
    pasos.append("### Paso 2: Regla de Sarrus para matriz 3x3")
    a, b, c = matriz[0,0], matriz[0,1], matriz[0,2]
    d, e, f = matriz[1,0], matriz[1,1], matriz[1,2]
    g, h, i = matriz[2,0], matriz[2,1], matriz[2,2]
    
    pasos.append("Matriz:")
    pasos.append(f"```\n{matriz}\n```")
    pasos.append("Se suman los productos de las diagonales descendentes y se restan las ascendentes:")
    pasos.append(f"Diagonales descendentes: a·e·i, b·f·g, c·d·h -> {a}·{e}·{i}, {b}·{f}·{g}, {c}·{d}·{h}")
    pasos.append(f"Diagonales ascendentes: c·e·g, a·f·h, b·d·i -> {c}·{e}·{g}, {a}·{f}·{h}, {b}·{d}·{i}")
    det = (a*e*i + b*f*g + c*d*h) - (c*e*g + a*f*h + b*d*i)
    pasos.append(f"det(A) = ({a}·{e}·{i} + {b}·{f}·{g} + {c}·{d}·{h}) - ({c}·{e}·{g} + {a}·{f}·{h} + {b}·{d}·{i})")
    pasos.append(f"det(A) = {det}")
    return det, pasos

def calcular_determinante_expansion(matriz):
    pasos = []
    n = matriz.shape[0]
    
    if n == 1:
        resultado = matriz[0, 0]
        pasos.append(f"### Paso 2: Determinante de matriz 1x1")
        pasos.append(f"det(A) = {resultado}")
        return resultado, pasos
    
    elif n == 2:
        pasos.append("### Paso 2: Fórmula para matriz 2x2")
        pasos.append("det(A) = a₁₁·a₂₂ - a₁₂·a₂₁")
        det = matriz[0,0]*matriz[1,1] - matriz[0,1]*matriz[1,0]
        pasos.append(f"det(A) = {matriz[0,0]}·{matriz[1,1]} - {matriz[0,1]}·{matriz[1,0]}")
        pasos.append(f"det(A) = {det}")
        return det, pasos
    
    elif n == 3:
        # Mantener la explicación de expansión por cofactores también válida
        pasos.append("### Paso 2: Expansión por cofactores (3x3)")
        a, b, c = matriz[0,0], matriz[0,1], matriz[0,2]
        d, e, f = matriz[1,0], matriz[1,1], matriz[1,2]
        g, h, i = matriz[2,0], matriz[2,1], matriz[2,2]
        
        pasos.append("Matriz:")
        pasos.append(f"```\n{matriz}\n```")
        pasos.append("det(A) = a(ei - fh) - b(di - fg) + c(dh - eg)")
        det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
        pasos.append(f"det(A) = {a}·({e}·{i} - {f}·{h}) - {b}·({d}·{i} - {f}·{g}) + {c}·({d}·{h} - {e}·{g})")
        pasos.append(f"det(A) = {det}")
        return det, pasos
    
    else:
        # Para matrices mayores a 3x3, usar expansión por cofactores
        pasos.append("### Paso 2: Expansión por cofactores")
        # Usar numpy para el cálculo (pero mostrar el proceso)
        det = np.linalg.det(matriz)
        pasos.append(f"Para matrices de {n}x{n}, el cálculo se realiza mediante expansión por cofactores:")
        pasos.append(f"det(A) = ∑ (-1)ⁱ⁺ʲ · aᵢⱼ · det(Mᵢⱼ)")
        pasos.append(f"Resultado: det(A) = {det}")
        return det, pasos

def calcular_determinante_reduccion(matriz):
    pasos = []
    n = matriz.shape[0]
    
    pasos.append("### Paso 2: Reducción por filas a forma triangular")
    # Crear una copia de la matriz
    mat = matriz.astype(float).copy()
    det = 1.0
    
    pasos.append("Matriz original:")
    pasos.append(f"```\n{mat}\n```")
    
    for i in range(n):
        # Pivoteo parcial si el elemento diagonal es cero
        if abs(mat[i,i]) < 1e-10:
            for j in range(i+1, n):
                if abs(mat[j,i]) > 1e-10:
                    mat[[i,j]] = mat[[j,i]]
                    det *= -1
                    pasos.append(f"Intercambiar fila {i+1} con fila {j+1}:")
                    pasos.append(f"```\n{mat}\n```")
                    break
        
        pivot = mat[i,i]
        if abs(pivot) < 1e-10:
            det = 0
            pasos.append(f"Pivote cero en posición ({i+1},{i+1}), determinante = 0")
            break
        
        det *= pivot
        pasos.append(f"Pivote en posición ({i+1},{i+1}): {pivot}")
        
        # Hacer ceros debajo del pivote
        for j in range(i+1, n):
            factor = mat[j,i] / pivot
            mat[j,i:] = mat[j,i:] - factor * mat[i,i:]
            pasos.append(f"F{j+1} = F{j+1} - ({factor})·F{i+1}:")
            pasos.append(f"```\n{mat}\n```")
    
    pasos.append(f"### Paso 3: Determinante como producto de la diagonal")
    diagonal = [mat[i,i] for i in range(n)]
    pasos.append(f"Diagonal: {diagonal}")
    pasos.append(f"det(A) = {' · '.join([f'{x:.2f}' for x in diagonal])} = {det}")
    
    return det, pasos

def calcular_inversa_adjunta(matriz):
    pasos = []
    n = matriz.shape[0]
    
    pasos.append("### Paso 3: Método de la matriz adjunta")
    
    # Calcular determinante
    det = np.linalg.det(matriz)
    pasos.append(f"det(A) = {det}")
    
    # Calcular matriz de cofactores
    pasos.append("### Paso 4: Calcular matriz de cofactores")
    cofactores = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Menor (submatriz sin fila i y columna j)
            menor = np.delete(np.delete(matriz, i, axis=0), j, axis=1)
            cofactor = ((-1) ** (i + j)) * np.linalg.det(menor)
            cofactores[i, j] = cofactor
    
    pasos.append("Matriz de cofactores:")
    pasos.append(f"```\n{cofactores}\n```")
    
    # Matriz adjunta es la transpuesta de la matriz de cofactores
    pasos.append("### Paso 5: Calcular matriz adjunta (transpuesta de cofactores)")
    adjunta = cofactores.T
    pasos.append("Matriz adjunta:")
    pasos.append(f"```\n{adjunta}\n```")
    
    # Matriz inversa = adjunta / determinante
    pasos.append("### Paso 6: Calcular inversa = adjunta / determinante")
    inversa = adjunta / det
    pasos.append("A⁻¹ = adj(A) / det(A):")
    pasos.append(f"```\n{inversa}\n```")
    
    return inversa, pasos

def calcular_inversa_gauss_jordan(matriz):
    pasos = []
    n = matriz.shape[0]
    
    pasos.append("### Paso 3: Método de Gauss-Jordan")
    
    # Crear matriz aumentada [A|I]
    identidad = np.eye(n)
    aumentada = np.hstack([matriz.astype(float), identidad])
    
    pasos.append("Matriz aumentada [A|I]:")
    pasos.append(f"```\n{aumentada}\n```")
    
    # Eliminación gaussiana
    for i in range(n):
        # Pivoteo parcial
        if abs(aumentada[i,i]) < 1e-10:
            for j in range(i+1, n):
                if abs(aumentada[j,i]) > 1e-10:
                    aumentada[[i,j]] = aumentada[[j,i]]
                    pasos.append(f"Intercambiar fila {i+1} con fila {j+1}:")
                    pasos.append(f"```\n{aumentada}\n```")
                    break
        
        # Hacer el pivote igual a 1
        pivot = aumentada[i,i]
        aumentada[i] = aumentada[i] / pivot
        pasos.append(f"F{i+1} = F{i+1} / {pivot}:")
        pasos.append(f"```\n{aumentada}\n```")
        
        # Hacer ceros en la columna i
        for j in range(n):
            if j != i:
                factor = aumentada[j,i]
                aumentada[j] = aumentada[j] - factor * aumentada[i]
                pasos.append(f"F{j+1} = F{j+1} - ({factor})·F{i+1}:")
                pasos.append(f"```\n{aumentada}\n```")
    
    # La inversa está en la parte derecha de la matriz aumentada
    inversa = aumentada[:, n:]
    pasos.append("### Paso 4: Matriz inversa obtenida:")
    pasos.append(f"```\n{inversa}\n```")
    
    return inversa, pasos

def calcular_rango_pasos(matriz):
    pasos = []
    mat = matriz.astype(float).copy()
    m, n = mat.shape
    
    pasos.append("### Paso 2: Eliminación gaussiana para calcular rango")
    pasos.append("Matriz original:")
    pasos.append(f"```\n{mat}\n```")
    
    rango = 0
    i, j = 0, 0
    
    while i < m and j < n:
        # Encontrar pivote en columna j
        pivot_fila = -1
        for k in range(i, m):
            if abs(mat[k,j]) > 1e-10:
                pivot_fila = k
                break
        
        if pivot_fila == -1:
            j += 1
            continue
        
        # Intercambiar filas si es necesario
        if pivot_fila != i:
            mat[[i, pivot_fila]] = mat[[pivot_fila, i]]
            pasos.append(f"Intercambiar fila {i+1} con fila {pivot_fila+1}:")
            pasos.append(f"```\n{mat}\n```")
        
        # Hacer ceros debajo del pivote
        pivot = mat[i,j]
        for k in range(i+1, m):
            factor = mat[k,j] / pivot
            mat[k,j:] = mat[k,j:] - factor * mat[i,j:]
        
        pasos.append(f"Eliminar debajo del pivote en ({i+1},{j+1}):")
        pasos.append(f"```\n{mat}\n```")
        
        rango += 1
        i += 1
        j += 1
    
    pasos.append(f"### Paso 3: Rango calculado")
    pasos.append(f"Rango = {rango} (número de filas linealmente independientes)")
    
    return rango, pasos

# Para usar la función principal en Streamlit
if __name__ == "__main__":
    operacion_una_matriz()