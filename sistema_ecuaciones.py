import streamlit as st
import numpy as np
import sympy as sp
import re
from sympy import Matrix, symbols

def mostrar_sistema_ecuaciones():
    st.header(":blue[Sistema de Ecuaciones Lineales]", divider="blue")
    
    # Configuración del sistema
    col_config, col_info = st.columns(2)
    
    with col_config:
        st.subheader("Configuración del Sistema")
        
        # Solo número de ecuaciones
        num_ecuaciones = st.number_input("Número de ecuaciones", min_value=1, max_value=10, value=3, step=1)
        
        # Asumimos mismo número de variables que ecuaciones
        num_variables = num_ecuaciones
        
        # Cambiar a "Por celdas" y "Por texto"
        tipo_ingreso = st.radio(
            "Modo de ingreso:",
            ["Por celdas", "Por texto"],
            horizontal=True,
            key="tipo_ingreso_sistema"
        )
    
    with col_info:
        st.success("""
        **INSTRUCCIONES**
        1. Ingresa el número de ecuaciones
        2. Elige el modo de ingreso (por celdas o por texto)
        3. Ingresa los coeficientes del sistema
        4. Selecciona el método de solución
        5. Observa la solución y el paso a paso
        """)
    
    # Limpiar resultados anteriores cuando cambia el tipo de ingreso
    if "ultimo_tipo_ingreso" not in st.session_state:
        st.session_state.ultimo_tipo_ingreso = tipo_ingreso
    
    if st.session_state.ultimo_tipo_ingreso != tipo_ingreso:
        # Limpiar resultados anteriores
        if "resultado_sistema" in st.session_state:
            del st.session_state["resultado_sistema"]
        if "metodo_sistema" in st.session_state:
            del st.session_state["metodo_sistema"]
        if "decimales" in st.session_state:
            del st.session_state["decimales"]
        st.session_state.ultimo_tipo_ingreso = tipo_ingreso
        st.rerun()
    
    # Ingreso del sistema de ecuaciones
    # Contenedor con borde
    with st.container(border=True):
        st.subheader(":blue[Ingreso del Sistema de Ecuaciones]")

        if tipo_ingreso == "Por celdas":
            sistema = crear_sistema_por_celdas(num_ecuaciones, num_variables)
        else:
            sistema = crear_sistema_por_texto(num_ecuaciones, num_variables)

    
    # Métodos de solución
    st.subheader(":blue[Método de Solución]")
    
    col_metodo, col_resultado = st.columns(2,border=True)
    
    with col_metodo:
        st.markdown("**Selecciona el método de solución:**")
        
        # Definir métodos disponibles basado en el número de ecuaciones
        metodos_base = ["Selecciona un método", "Eliminación Gaussiana", "Gauss-Jordan", "Método de Sustitución"]
        
        # Agregar métodos que solo funcionan bien para sistemas pequeños (hasta 3 ecuaciones)
        if num_ecuaciones <= 3:
            metodos_disponibles = metodos_base + ["Matriz Inversa", "Regla de Cramer"]
        else:
            metodos_disponibles = metodos_base
            st.info("💡 Para más de 3 ecuaciones, solo se muestran métodos eficientes")
        
        metodo = st.selectbox(
            "Método:",
            metodos_disponibles,
            key="metodo_select"
        )
        
        # Mostrar advertencia para métodos ineficientes en sistemas grandes
        if num_ecuaciones > 3 and metodo in ["Matriz Inversa", "Regla de Cramer"]:
            st.warning("⚠️ Este método puede ser ineficiente para sistemas grandes")
        
        # Configuración adicional
        st.markdown("---")
        mostrar_decimales = st.checkbox("Mostrar números decimales", value=True, key="mostrar_decimales")
        if mostrar_decimales:
            decimales = st.number_input("Número de decimales", min_value=1, max_value=10, value=2, step=1, key="num_decimales")
        else:
            decimales = None
        
        # Botón de resolver
        if st.button("Resolver Sistema", type="primary", use_container_width=True, key="resolver_sistema"):
            if metodo == "Selecciona un método":
                st.warning("Por favor selecciona un método de solución primero.")
            elif sistema is not None:
                resultado = resolver_sistema(sistema, metodo, decimales)
                st.session_state["resultado_sistema"] = resultado
                st.session_state["metodo_sistema"] = metodo
                st.session_state["decimales"] = decimales
            else:
                st.error("Error: No se pudo crear el sistema de ecuaciones.")
    
    with col_resultado:
        st.subheader(":blue[Resultado]")
        if "resultado_sistema" in st.session_state:
            mostrar_resultado_sistema(
                st.session_state["resultado_sistema"],
                st.session_state["metodo_sistema"],
                st.session_state.get("decimales", 2)
            )
    
    # SECCIÓN DE PASO A PASO
    if "resultado_sistema" in st.session_state and "pasos" in st.session_state["resultado_sistema"]:
        st.markdown("---")
        st.subheader("🧮 Paso a Paso de la Solución")
        
        with st.expander("**Ver procedimiento completo**", expanded=True):
            pasos = st.session_state["resultado_sistema"]["pasos"]
            for i, paso in enumerate(pasos, 1):
                if isinstance(paso, dict):
                    st.markdown(f"**Paso {i}: {paso['titulo']}**")
                    
                    if 'descripcion' in paso:
                        st.write(paso['descripcion'])
                    
                    if 'contenido' in paso:
                        if isinstance(paso['contenido'], str) and ('$$' in paso['contenido'] or '\\' in paso['contenido']):
                            st.latex(paso['contenido'])
                        else:
                            st.write(paso['contenido'])
                    
                    if 'matriz' in paso:
                        st.write("Matriz aumentada:")
                        # Formatear la matriz para mejor visualización
                        matriz_formateada = paso['matriz'].copy()
                        n_vars = matriz_formateada.shape[1] - 1
                        
                        # Crear DataFrame con nombres de columnas apropiados
                        columnas = [f"x_{{{i+1}}}" for i in range(n_vars)] + ["b"]
                        filas = [f"Ecuación {i+1}" for i in range(matriz_formateada.shape[0])]
                        
                        # Redondear valores para mejor visualización
                        matriz_formateada = np.round(matriz_formateada, 4)
                        
                        st.dataframe(matriz_formateada, 
                                   column_config={f"col_{i}": col for i, col in enumerate(columnas)},
                                   use_container_width=True)
                        
                else:
                    st.write(f"**Paso {i}:** {paso}")

def crear_sistema_por_celdas(num_ecuaciones, num_variables):
    st.markdown("**Ingresa los coeficientes del sistema:**")
    
    # Botones para limpiar y generar aleatorios
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🧹 Limpiar todo", key="limpiar_celdas"):
            for i in range(num_ecuaciones):
                for j in range(num_variables):
                    key = f"coef_celda_{i}_{j}"
                    if key in st.session_state:
                        del st.session_state[key]
                key_indep = f"indep_celda_{i}"
                if key_indep in st.session_state:
                    del st.session_state[key_indep]
            if "resultado_sistema" in st.session_state:
                del st.session_state["resultado_sistema"]
            st.rerun()
    
    with col_btn2:
        if st.button("🎲 Generar aleatorio", key="aleatorio_celdas"):
            for i in range(num_ecuaciones):
                for j in range(num_variables):
                    key = f"coef_celda_{i}_{j}"
                    st.session_state[key] = np.random.randint(-5, 6)
                key_indep = f"indep_celda_{i}"
                st.session_state[key_indep] = np.random.randint(-10, 11)
            if "resultado_sistema" in st.session_state:
                del st.session_state["resultado_sistema"]
            st.rerun()
    
    # Matriz de coeficientes y vector independiente
    matriz_coef = np.zeros((num_ecuaciones, num_variables))
    vector_indep = np.zeros(num_ecuaciones)
    variables = [f"x_{{{i+1}}}" for i in range(num_variables)]
    
    for i in range(num_ecuaciones):
        st.markdown(f"**Ecuación {i+1}:**")
        cols = st.columns(num_variables * 2 + 1)
        
        for j in range(num_variables):
            with cols[j * 2]:
                key = f"coef_celda_{i}_{j}"
                if key not in st.session_state:
                    st.session_state[key] = 0
                
                value = st.session_state[key]
                display_value = int(value) if value == int(value) else value
                
                coef = st.number_input(
                    f"Coeficiente de {variables[j]}",
                    value=float(display_value),
                    key=key,
                    format="%g",
                    label_visibility="collapsed"
                )
                matriz_coef[i, j] = coef
            
            if j < num_variables - 1:
                with cols[j * 2 + 1]:
                    st.markdown("<div style='text-align: center; margin-top: 1.5rem;'>+</div>", 
                               unsafe_allow_html=True)
        
        with cols[num_variables * 2 - 1]:
            st.markdown("<div style='text-align: center; margin-top: 1.5rem;'>=</div>", 
                       unsafe_allow_html=True)
        
        with cols[num_variables * 2]:
            key_indep = f"indep_celda_{i}"
            if key_indep not in st.session_state:
                st.session_state[key_indep] = 0
            
            value_indep = st.session_state[key_indep]
            display_value_indep = int(value_indep) if value_indep == int(value_indep) else value_indep
            
            vector_indep[i] = st.number_input(
                f"Término independiente ecuación {i+1}",
                value=float(display_value_indep),
                key=key_indep,
                format="%g",
                label_visibility="collapsed"
            )
    
    return {
        "coeficientes": matriz_coef, 
        "independientes": vector_indep,
        "variables": variables,
        "tipo": "numerico"
    }

def crear_sistema_por_texto(num_ecuaciones, num_variables):
    st.markdown("**Ingresa las ecuaciones en formato texto:**")
    st.info("Ejemplo: `2x + 3y - z = 5` o `-3a + 5b - 2c = 10`")
    
    if st.button("🧹 Limpiar ecuaciones", key="limpiar_texto"):
        for i in range(num_ecuaciones):
            key = f"ecuacion_texto_{i}"
            if key in st.session_state:
                del st.session_state[key]
        if "resultado_sistema" in st.session_state:
            del st.session_state["resultado_sistema"]
        st.rerun()
    
    ecuaciones_texto = []
    for i in range(num_ecuaciones):
        key = f"ecuacion_texto_{i}"
        if key not in st.session_state:
            if num_variables <= 3:
                vars_default = ['x', 'y', 'z'][:num_variables]
                default_eq = " + ".join([f"{i+1}{v}" for v in vars_default]) + f" = {i+1}"
            else:
                vars_default = [f"x{j+1}" for j in range(num_variables)]
                default_eq = " + ".join([f"{i+1}{v}" for v in vars_default]) + f" = {i+1}"
            st.session_state[key] = default_eq
        
        ecuacion = st.text_input(
            f"Ecuación {i+1}",
            value=st.session_state[key],
            key=key,
            placeholder=f"Ej: 2x + 3y - z = {i+1}"
        )
        ecuaciones_texto.append(ecuacion)
    
    try:
        return parsear_ecuaciones_simple(ecuaciones_texto, num_variables)
    except Exception as e:
        st.error(f"Error al procesar las ecuaciones: {e}")
        return None

def parsear_ecuaciones_simple(ecuaciones_texto, num_variables):
    ecuaciones_validas = [eq for eq in ecuaciones_texto if eq.strip()]
    
    if not ecuaciones_validas:
        st.error("No se ingresaron ecuaciones válidas.")
        return None
    
    variables_comunes = ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f']
    variables_usadas = variables_comunes[:num_variables]
    
    matriz_coef = np.zeros((len(ecuaciones_validas), num_variables))
    vector_indep = np.zeros(len(ecuaciones_validas))
    
    for i, ecuacion_texto in enumerate(ecuaciones_validas):
        try:
            ecuacion = ecuacion_texto.replace(' ', '').lower()
            
            if '=' in ecuacion:
                partes = ecuacion.split('=')
                lado_izq = partes[0]
                lado_der = partes[1] if len(partes) > 1 else '0'
            else:
                lado_izq = ecuacion
                lado_der = '0'
            
            for j, var in enumerate(variables_usadas):
                patron = r'([+-]?[0-9.]*)' + var + r'([+-]|$|)'
                coincidencias = re.findall(patron, lado_izq)
                
                coef_total = 0
                for coef_str, _ in coincidencias:
                    if coef_str in ['', '+']:
                        coef_total += 1
                    elif coef_str == '-':
                        coef_total -= 1
                    else:
                        if coef_str.startswith('+'):
                            coef_num = coef_str[1:] if coef_str[1:] else '1'
                        elif coef_str.startswith('-'):
                            coef_num = '-' + (coef_str[1:] if coef_str[1:] else '1')
                        else:
                            coef_num = coef_str
                        
                        try:
                            coef_total += float(coef_num) if coef_num else 1.0
                        except:
                            coef_total += 1.0
                
                matriz_coef[i, j] = coef_total
            
            try:
                term_indep = float(lado_der) if lado_der else 0
            except:
                term_indep = 0
            
            constantes_izq = re.findall(r'([+-]?[0-9.]+)(?![a-zA-Z])', lado_izq)
            for const in constantes_izq:
                if const.startswith('+'):
                    term_indep -= float(const[1:]) if const[1:] else 1
                elif const.startswith('-'):
                    term_indep += float(const[1:]) if const[1:] else 1
                else:
                    term_indep -= float(const) if const else 0
            
            vector_indep[i] = term_indep
            
        except Exception as e:
            st.error(f"Error en ecuación {i+1}: '{ecuacion_texto}' - {str(e)}")
            return None
    
    st.success("✅ **Sistema parseado correctamente**")
    st.write(f"Variables detectadas: {variables_usadas}")
    
    variables_latex = [f"{v}_{{{i+1}}}" for i, v in enumerate(variables_usadas)]
    
    return {
        "coeficientes": matriz_coef, 
        "independientes": vector_indep,
        "variables": variables_latex,
        "ecuaciones_texto": ecuaciones_texto,
        "tipo": "texto"
    }

def resolver_sistema(sistema, metodo, decimales=None):
    try:
        A = sistema["coeficientes"]
        b = sistema["independientes"]
        variables = sistema["variables"]
        pasos = []
        
        det_A = np.linalg.det(A)
        if abs(det_A) < 1e-10:
            return {"error": "La matriz de coeficientes es incompatible. El sistema no tiene solución.", "pasos": pasos}
        
        # Mostrar el sistema de manera clara
        ecuaciones_latex = mostrar_ecuaciones_latex_claro(A, b, variables)
        pasos.append({
            "titulo": "Sistema de ecuaciones original", 
            #"descripcion": "Este es el sistema de ecuaciones que vamos a resolver:",
            "contenido": f"$${ecuaciones_latex}$$"
        })
        
        if metodo == "Eliminación Gaussiana":
            return resolver_gaussiana(A, b, variables, pasos, decimales)
        elif metodo == "Gauss-Jordan":
            return resolver_gauss_jordan(A, b, variables, pasos, decimales)
        elif metodo == "Método de Sustitución":
            return resolver_sustitucion(A, b, variables, pasos, decimales)
        elif metodo == "Matriz Inversa":
            return resolver_matriz_inversa(A, b, variables, pasos, decimales)
        elif metodo == "Regla de Cramer":
            return resolver_cramer(A, b, variables, pasos, decimales)
        else:
            return {"error": f"Método '{metodo}' no reconocido", "pasos": pasos}
    
    except Exception as e:
        return {"error": f"Error al resolver el sistema: {str(e)}", "pasos": pasos}

def resolver_gaussiana(A, b, variables, pasos, decimales):
    n = len(b)
    # Crear una copia para no modificar la original
    Ab = np.hstack([A.copy(), b.reshape(-1, 1)])
    
    pasos.append({
        "titulo": "Matriz aumentada inicial",
        "descripcion": "Creamos la matriz aumentada [A|b]:",
        "matriz": Ab.copy()
    })
    
    # Eliminación hacia adelante
    for i in range(n):
        # Pivoteo parcial
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]
            pasos.append({
                "titulo": f"Intercambio de filas {i+1} y {max_row+1}",
                "descripcion": f"Intercambiamos las filas para tener el mayor pivote en la posición ({i+1},{i+1})",
                "matriz": Ab.copy()
            })
        
        # Eliminación
        for j in range(i + 1, n):
            if abs(Ab[i, i]) > 1e-10:  # Evitar división por cero
                factor = Ab[j, i] / Ab[i, i]
                Ab[j] = Ab[j] - factor * Ab[i]
                pasos.append({
                    "titulo": f"Eliminación en fila {j+1}",
                    "descripcion": f"F_{{{j+1}}} = F_{{{j+1}}} - ({factor:.4f}) × F_{{{i+1}}}",
                    "matriz": Ab.copy()
                })
    
    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]
        x[i] = round(x[i], 10)
        pasos.append({
            "titulo": f"Sustitución hacia atrás para {variables[i]}",
            "descripcion": f"Despejamos {variables[i]} de la ecuación {i+1}",
            "contenido": f"{variables[i]} = {x[i]:.6f}"
        })
    
    return {"solucion": x, "metodo": "Eliminación Gaussiana", "pasos": pasos}

def resolver_gauss_jordan(A, b, variables, pasos, decimales):
    n = len(b)
    Ab = np.hstack([A.copy(), b.reshape(-1, 1)])
    
    pasos.append({
        "titulo": "Matriz aumentada inicial",
        "descripcion": "Creamos la matriz aumentada [A|b]:",
        "matriz": Ab.copy()
    })
    
    for i in range(n):
        # Pivoteo
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]
            pasos.append({
                "titulo": f"Pivoteo: intercambio fila {i+1} con fila {max_row+1}",
                "matriz": Ab.copy()
            })
        
        # Normalizar el pivote a 1
        pivot = Ab[i, i]
        if abs(pivot) > 1e-10:
            Ab[i] = Ab[i] / pivot
            pasos.append({
                "titulo": f"Normalizar fila {i+1}",
                "descripcion": f"Dividimos la fila {i+1} por el pivote {pivot:.4f}",
                "matriz": Ab.copy()
            })
        
        # Eliminación en todas las demás filas
        for j in range(n):
            if j != i and abs(Ab[j, i]) > 1e-10:
                factor = Ab[j, i]
                Ab[j] = Ab[j] - factor * Ab[i]
                pasos.append({
                    "titulo": f"Eliminar variable en fila {j+1}",
                    "descripcion": f"F_{{{j+1}}} = F_{{{j+1}}} - ({factor:.4f}) × F_{{{i+1}}}",
                    "matriz": Ab.copy()
                })
    
    # La solución está en la última columna
    x = Ab[:, -1].copy()
    x = np.round(x, 10)
    
    pasos.append({
        "titulo": "Matriz identidad obtenida",
        "descripcion": "La matriz ahora está en forma de identidad, la solución está en la última columna",
        "matriz": Ab.copy()
    })
    
    return {"solucion": x, "metodo": "Gauss-Jordan", "pasos": pasos}

def resolver_sustitucion(A, b, variables, pasos, decimales):
    n = len(b)
    x = np.zeros(n)
    
    # Verificar si la matriz es triangular
    es_triangular_superior = np.allclose(A, np.triu(A))
    es_triangular_inferior = np.allclose(A, np.tril(A))
    
    if not (es_triangular_superior or es_triangular_inferior):
        pasos.append({
            "titulo": "Advertencia", 
            "descripcion": "El sistema no es triangular. Aplicaremos eliminación gaussiana primero para triangularizarlo."
        })
        
        # Triangularizar el sistema
        Ab = np.hstack([A.copy(), b.reshape(-1, 1)])
        for i in range(n):
            for j in range(i + 1, n):
                if abs(Ab[i, i]) > 1e-10:
                    factor = Ab[j, i] / Ab[i, i]
                    Ab[j] = Ab[j] - factor * Ab[i]
        
        A = Ab[:, :-1]
        b = Ab[:, -1]
        pasos.append({
            "titulo": "Sistema triangularizado",
            "matriz": Ab.copy()
        })
        es_triangular_superior = True
    
    if es_triangular_superior:
        # Sustitución hacia atrás
        pasos.append({
            "titulo": "Sustitución hacia atrás",
            "descripcion": "Resolvemos de la última ecuación a la primera"
        })
        for i in range(n-1, -1, -1):
            x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
            x[i] = round(x[i], 10)
            pasos.append({
                "titulo": f"Sustitución para {variables[i]}",
                "descripcion": f"Despejamos {variables[i]} de la ecuación {i+1}",
                "contenido": f"{variables[i]} = {x[i]:.6f}"
            })
    else:
        # Sustitución hacia adelante
        pasos.append({
            "titulo": "Sustitución hacia adelante",
            "descripcion": "Resolvemos de la primera ecuación a la última"
        })
        for i in range(n):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i])) / A[i, i]
            x[i] = round(x[i], 10)
            pasos.append({
                "titulo": f"Sustitución para {variables[i]}",
                "descripcion": f"Despejamos {variables[i]} de la ecuación {i+1}",
                "contenido": f"{variables[i]} = {x[i]:.6f}"
            })
    
    return {"solucion": x, "metodo": "Método de Sustitución", "pasos": pasos}

def resolver_matriz_inversa(A, b, variables, pasos, decimales):
    det_A = np.linalg.det(A)
    pasos.append({
        "titulo": "Cálculo del determinante", 
        "descripcion": "Calculamos el determinante de la matriz A para verificar si es invertible",
        "contenido": f"\\det(A) = {det_A:.6f}"
    })
    
    if abs(det_A) < 1e-10:
        return {"error": "La matriz no es invertible (determinante ≈ 0)", "pasos": pasos}
    
    A_inv = np.linalg.inv(A)
    pasos.append({
        "titulo": "Matriz inversa calculada", 
        "descripcion": "Calculamos la inversa de la matriz A",
        "matriz": np.hstack([A_inv, np.zeros((A_inv.shape[0], 1))])  # Para mantener formato consistente
    })
    
    x = A_inv @ b
    x = np.round(x, 10)
    pasos.append({
        "titulo": "Solución por multiplicación", 
        "descripcion": "Multiplicamos la matriz inversa por el vector de términos independientes",
        "contenido": f"x = A^{{-1}} \\cdot b"
    })
    
    return {"solucion": x, "metodo": "Matriz Inversa", "pasos": pasos}

def resolver_cramer(A, b, variables, pasos, decimales):
    det_A = np.linalg.det(A)
    pasos.append({
        "titulo": "Determinante de la matriz principal", 
        "descripcion": "Calculamos el determinante de la matriz A",
        "contenido": f"\\det(A) = {det_A:.6f}"
    })
    
    if abs(det_A) < 1e-10:
        return {"error": "El sistema no tiene solución única (determinante = 0)", "pasos": pasos}
    
    n = len(b)
    x = np.zeros(n)
    
    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b
        det_A_i = np.linalg.det(A_i)
        x[i] = det_A_i / det_A
        x[i] = round(x[i], 10)
        pasos.append({
            "titulo": f"Cálculo de {variables[i]}", 
            "descripcion": f"Reemplazamos la columna {i+1} de A con el vector b y calculamos el determinante",
            "contenido": f"{variables[i]} = \\frac{{\\det(A_{{{i+1}}})}}{{\\det(A)}} = \\frac{{{det_A_i:.6f}}}{{{det_A:.6f}}} = {x[i]:.6f}"
        })
    
    return {"solucion": x, "metodo": "Regla de Cramer", "pasos": pasos}

def mostrar_ecuaciones_latex_claro(A, b, variables):
    """Muestra el sistema de ecuaciones de manera clara y entendible"""
    n = len(b)
    ecuaciones = []
    
    for i in range(n):
        terminos = []
        for j in range(len(variables)):
            coef = A[i, j]
            # Solo mostrar términos con coeficiente no cero
            if abs(coef) > 1e-10:
                if abs(coef - 1) < 1e-10:
                    term_str = f"{variables[j]}"
                elif abs(coef + 1) < 1e-10:
                    term_str = f"-{variables[j]}"
                elif coef > 0:
                    # Para el primer término, no mostrar el signo +
                    if not terminos:
                        term_str = f"{coef:.2f}{variables[j]}"
                    else:
                        term_str = f"+ {abs(coef):.2f}{variables[j]}"
                else:
                    term_str = f"- {abs(coef):.2f}{variables[j]}"
                terminos.append(term_str)
        
        # Si no hay términos, mostrar 0
        if not terminos:
            ecuacion_str = "0"
        else:
            ecuacion_str = ' '.join(terminos)
        
        ecuaciones.append(f"{ecuacion_str} = {b[i]:.2f}")
    
    return "\\\\".join(ecuaciones)

def mostrar_resultado_sistema(resultado, metodo, decimales):
    if "error" in resultado:
        st.error(resultado["error"])
        return
    
    st.success(f"**Solución del sistema ({metodo}):**")
    solucion = resultado["solucion"]
    
    if decimales is None:
        decimales = 2
    
    for i, valor in enumerate(solucion):
        valor_limpio = round(valor, decimales)
        if abs(valor_limpio - round(valor_limpio)) < 1e-10:
            valor_formateado = str(int(round(valor_limpio)))
        else:
            valor_formateado = f"{valor_limpio:.{decimales}f}"
        
        st.latex(f"x_{{{i+1}}} = {valor_formateado}")