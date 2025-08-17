import streamlit as st        # Para crear aplicaciones web interactivas en Python
import numpy as np            # Manejo de arreglos y operaciones numéricas
import matplotlib.pyplot as plt  # Gráficos estáticos
import plotly.express as px   # Gráficos interactivos
from PIL import Image         # Manipulación de imágenes
import io                     # Entrada/salida en memoria (streams)
import pandas as pd           # Manejo de datos tabulares (DataFrames)

st.set_page_config(page_title="Métodos Numéricos", layout="wide", page_icon="smile")

algorithm, inicio, analitico, ejemplos, aplicaciones, conclusion = st.tabs(
    ["Algorithm", "Inicio", "Analítico vs Numérico", "Ejemplos", "Aplicaciones", "Conclusión"]
)

with algorithm:
    st.header("Integrantes del grupo - Developer Algorithm")
    st.markdown("""
    ### Miembros:
    - Carla Valeria Encinas Cano - Lider de Grupo
    - Victor Grover Asturizaga Plata
    - Milenka Melody Chuquimia Osco
    - Jair Fabricio Vera Pozo
    - Isaac Emannuel Escobar Rios
    """)

def mostrar_inicio():
    st.header("Introducción a los Métodos Numéricos")
    
    st.markdown("""
    Los **métodos numéricos** son técnicas para obtener soluciones aproximadas a problemas matemáticos cuando no se pueden resolver con fórmulas exactas o son demasiado complejos para una solución analítica.

    ---
    
    ## ¿Por qué usar métodos numéricos?
    - Muchas funciones no tienen raíces expresables en forma cerrada.
    - Problemas de integración, derivación, resolución de sistemas, ecuaciones diferenciales, etc., requieren aproximaciones.
    - Son esenciales en ingeniería, física, finanzas, ciencias de la computación y más.
    
    ---
    
    ## Características de los métodos numéricos
    - **Iterativos:** comienzan con una o más estimaciones iniciales y mejoran la solución paso a paso.
    - **Control de error:** miden la diferencia entre aproximaciones para asegurar precisión.
    - **Convergencia:** bajo ciertas condiciones, garantizan acercarse a la solución real.
    
    ---
    
    ## Métodos numéricos para encontrar raíces de funciones
    
    ### Método de Newton-Raphson
    Se basa en la aproximación lineal de la función alrededor de un punto \( x_i \):
    """)
    
    st.latex(r"x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)}")
    
    st.markdown("""
    Es rápido y eficiente si la derivada es conocida y la estimación inicial está cerca de la raíz.
    
    ---
    
    ### Método de la Bisección
    Utiliza el teorema del valor intermedio. Requiere un intervalo \([a,b]\) donde la función cambie de signo (\(f(a) \cdot f(b) < 0\)).
    
    Se divide el intervalo a la mitad repetidamente:
    """)
    
    st.latex(r"c = \frac{a + b}{2}")
    
    st.markdown("""
    Se selecciona el subintervalo donde la función cambia de signo, hasta que el error sea menor que la tolerancia.
    
    Es robusto pero más lento.
    
    ---
    """)
    st.markdown("### Método de la Secante")
    st.markdown("Similar a Newton-Raphson pero no necesita la derivada explícita.")

    st.markdown("Utiliza dos puntos:")
    st.latex(r"x_{i-1} \quad \text{y} \quad x_i")
    st.markdown("para aproximar la derivada:")

    st.latex(r"x_{i+1} = x_i - f(x_i) \frac{x_i - x_{i-1}}{f(x_i) - f(x_{i-1})}")

    st.markdown("""
    Combina velocidad y no dependencia en la derivada, pero puede ser menos estable.
    """)

    st.latex(r"x_{i+1} = x_i - f(x_i) \frac{x_i - x_{i-1}}{f(x_i) - f(x_{i-1})}")
    
    st.markdown("""
    Combina velocidad y no dependencia en la derivada, pero puede ser menos estable.
    
    ---
    
    ## ¿Qué encontrarás en esta aplicación?
    - **Analítico vs Numérico:** Compararemos la solución exacta con métodos aproximados para entender sus ventajas y limitaciones.
    - **Ejemplos:** Exploraremos casos complejos donde los métodos analíticos no son viables, como simulaciones y ecuaciones diferenciales.
    - **Aplicaciones:** Demostraremos cómo los métodos numéricos se usan en simulación de proyectiles, optimización en IA, compresión de imágenes usando SVD y simulaciones financieras con Monte Carlo.
    - **Resultados:** Visualización y descarga de datos para análisis posterior.
    - **Conclusión:** Recapitulación de conceptos clave y recursos para seguir aprendiendo.
    """)


with inicio:
    mostrar_inicio()


def mostrar_analitico_vs_numerico():
    st.header("Métodos Analíticos vs Métodos Numéricos")

    st.markdown("""
    ### ¿Qué es un método analítico?
    Es una técnica matemática para encontrar la solución exacta de un problema, usando fórmulas y transformaciones algebraicas o cálculo simbólico.

    ### ¿Qué es un método numérico?
    Es una técnica para obtener soluciones aproximadas mediante algoritmos computacionales, iterando hasta converger a una solución aceptable.

    **En esta sección podrás experimentar con varios métodos numéricos para encontrar raíces de funciones y comparar con la solución analítica si está disponible.**
    """)

    metodo = st.selectbox(
        "Elige el método numérico:",
        ["Newton-Raphson", "Bisección", "Secante"]
    )

    funcion_str = st.text_input("Función f(x) =", "x**3 - x - 2")
    derivada_str = st.text_input("Derivada f'(x) =", "3*x**2 - 1") if metodo == "Newton-Raphson" else None

    if metodo == "Bisección":
        a = st.number_input("Extremo izquierdo (a):", value=1)
        b = st.number_input("Extremo derecho (b):", value=2)
    else:
        x0 = st.number_input("Valor inicial x0:", value=1.5)
        if metodo == "Secante":
            x1 = st.number_input("Valor inicial x1:", value=2.0)

    tol = st.number_input("Tolerancia:", value=1e-6, format="%.1e")
    max_iter = st.number_input("Máx iteraciones:", value=50, step=1)

    try:
        from scipy.optimize import root_scalar # type: ignore
        f_analitico = lambda x: eval(funcion_str, {"x": x, "np": np})
        sol = root_scalar(f_analitico, bracket=[a if metodo=="Bisección" else 0, b if metodo=="Bisección" else 3], method='brentq')
        raiz_analitica = sol.root if sol.converged else None
    except:
        raiz_analitica = None

    if raiz_analitica is not None:
        st.write(f"Raíz analítica aproximada (brentq): {raiz_analitica:.6f}")
    else:
        st.write("No se encontró raíz analítica aproximada.")

    if st.button("Ejecutar método"):
        try:
            f = lambda x: eval(funcion_str, {"x": x, "np": np})

            if metodo == "Newton-Raphson":
                df = lambda x: eval(derivada_str, {"x": x, "np": np})
                iteraciones, errores = [], []
                xi = x0
                for i in range(max_iter):
                    x_new = xi - f(xi) / df(xi)
                    error = abs(x_new - xi)
                    iteraciones.append((i+1, xi, f(xi), error))
                    errores.append(error)
                    if error < tol:
                        break
                    xi = x_new
                resultado = xi

            elif metodo == "Bisección":
                iteraciones, errores = [], []
                ai, bi = a, b
                for i in range(max_iter):
                    c = (ai + bi) / 2
                    error = abs(bi - ai) / 2
                    iteraciones.append((i+1, c, f(c), error))
                    errores.append(error)
                    if f(ai) * f(c) < 0:
                        bi = c
                    else:
                        ai = c
                    if error < tol:
                        break
                resultado = c

            elif metodo == "Secante":
                iteraciones, errores = [], []
                xi, xi_1 = x0, x1
                for i in range(max_iter):
                    x_new = xi - f(xi) * (xi - xi_1) / (f(xi) - f(xi_1))
                    error = abs(x_new - xi)
                    iteraciones.append((i+1, xi, f(xi), error))
                    errores.append(error)
                    if error < tol:
                        break
                    xi_1, xi = xi, x_new
                resultado = xi

            st.subheader(f"Resultado método numérico: x ≈ {resultado:.6f}")

            if raiz_analitica is not None:
                st.write(f"Diferencia con raíz analítica: {abs(resultado - raiz_analitica):.6e}")

            df_tabla = pd.DataFrame(iteraciones, columns=["Iteración", "x", "f(x)", "Error"])
            st.dataframe(df_tabla)

            fig, ax = plt.subplots()
            ax.semilogy(range(1, len(errores)+1), errores, marker='o')
            ax.set_xlabel("Iteración")
            ax.set_ylabel("Error (escala log)")
            ax.set_title("Convergencia del método")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")
            
with analitico:
    mostrar_analitico_vs_numerico()

def mostrar_ejemplos():
    st.header("Ejemplos donde lo Analítico no es Práctico")

    st.markdown(r"""
    En muchos problemas reales, obtener una solución analítica exacta es imposible o demasiado complicado.  
    Los métodos numéricos permiten aproximar estas soluciones con alta precisión mediante algoritmos computacionales.

    **Algunos casos donde los métodos analíticos fallan o son muy difíciles:**  
    - Ecuaciones diferenciales no lineales complejas.  
    - Grandes simulaciones en física o ingeniería.  
    - Modelos estocásticos y procesos aleatorios.  
    - Integrales definidas de funciones sin primitiva elemental.
    """)

    st.subheader("Ejemplo 1: Crecimiento Poblacional con Ecuación Logística")

    st.markdown(r"""
    La ecuación diferencial logística es:  
    $$
        \frac{dP}{dt} = r P \left(1 - \frac{P}{K}\right)
    $$

    donde:  
    - \(P(t)\) es la población en el tiempo \(t\).  
    - \(r\) es la tasa de crecimiento.  
    - \(K\) es la capacidad máxima del ambiente (límite superior).
    """)

    r = st.slider("Tasa de crecimiento (r)", 0.1, 2.0, 0.5)
    K = st.number_input("Capacidad máxima (K)", min_value=10, max_value=1000, value=100)
    P0 = st.number_input("Población inicial (P0)", min_value=1, max_value=K, value=10)
    T = st.slider("Tiempo total de simulación", 1, 50, 20)

    dt = 0.1
    N = int(T / dt)
    t = [0]
    P = [P0]

    for i in range(1, N+1):
        P_new = P[-1] + r * P[-1] * (1 - P[-1] / K) * dt
        P.append(P_new)
        t.append(i * dt)

    fig, ax = plt.subplots()
    ax.plot(t, P, label="Población (P)")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Población")
    ax.set_title("Simulación numérica del crecimiento poblacional (modelo logístico)")
    ax.legend()
    st.pyplot(fig)

    st.markdown(r"""
    Aquí usamos el método de Euler explícito para aproximar la solución de la ecuación diferencial,  
    que no tiene una solución analítica simple para todos los casos, pero puede aproximarse numéricamente.
    """)

    st.subheader("Ejemplo 2: Aproximación numérica de una integral definida")

    st.markdown(r"""
    Consideremos la integral definida:  
    $$
        I = \int_0^1 e^{-x^2} \, dx
    $$

    que no tiene una primitiva elemental y debe aproximarse numéricamente.
    """)

    def f(x):
        return np.exp(-x**2)

    n = st.slider("Número de subintervalos para la aproximación", 10, 1000, 100)

    x_vals = np.linspace(0, 1, n+1)
    y_vals = f(x_vals)

    I_trap = (1/n) * (0.5*y_vals[0] + np.sum(y_vals[1:-1]) + 0.5*y_vals[-1])

    st.write(f"Integral aproximada con método del trapecio: {I_trap:.6f}")

    fig2, ax2 = plt.subplots()
    ax2.plot(x_vals, y_vals, label=r"$f(x) = e^{-x^2}$")
    ax2.fill_between(x_vals, 0, y_vals, color="skyblue", alpha=0.5)
    ax2.set_title("Área bajo la curva para la integral definida")
    st.pyplot(fig2)

    st.markdown(r"""
    Este método numérico divide el área bajo la curva en trapecios y suma sus áreas para aproximar la integral.
    """)


with ejemplos:
    mostrar_ejemplos()

with aplicaciones:
    st.header("Aplicaciones reales")

    sim_tab, ia_tab, img_tab, fin_tab= st.tabs(["Simulación", "Inteligencia Artificial", "Comprimir Imagen", "Finanzas"])

    with sim_tab:
        st.subheader("Simulación: trayectoría de proyectil (sin rozamiento)")

        st.markdown(r"""
        Las ecuaciones que describen la posición del proyectil en función del tiempo son:

        $$
        x(t) = v_0 \cos(\theta) \, t
        $$

        $$
        y(t) = v_0 \sin(\theta) \, t - \frac{1}{2} g t^2
        $$

        donde:

        - \(x₀\) es la velocidad inicial.  
        - \(θ\) es el ángulo de lanzamiento (en radianes).  
        - \(g = 9.81 \, m/s^2\) es la aceleración de la gravedad.  
        - \(t\) es el tiempo transcurrido.
        """)


        v0 = st.slider("Velocidad inicial (m/s)", 1, 200, 50)
        angle_deg = st.slider("Ángulo (grados)", 1, 89, 45)
        g = 9.81

        angle = np.deg2rad(angle_deg)

        t_flight = 2 * v0 * np.sin(angle) / g
        t = np.linspace(0, t_flight, 200)

        x = v0 * np.cos(angle) * t
        y = v0 * np.sin(angle) * t - 0.5 * g * t**2
        y = np.maximum(y, 0)

        alcance_max = x[-1]
        altura_max = np.max(y)

        st.write(f"**Alcance máximo:** {alcance_max:.2f} m")
        st.write(f"**Altura máxima:** {altura_max:.2f} m")

        fig, ax = plt.subplots()
        ax.plot(x, y, label="Trayectoria")
        ax.scatter(alcance_max, 0, color='red', label="Alcance máximo")
        ax.scatter(x[np.argmax(y)], altura_max, color='green', label="Altura máxima")
        ax.set_xlabel("Distancia (m)")
        ax.set_ylabel("Altura (m)")
        ax.set_title("Trayectoria de proyectil sin rozamiento")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


    with ia_tab:
        st.subheader("Optimización por Descenso de Gradiente (Ejemplo Simple)")

        st.markdown(r"""
        **¿Qué es el Descenso de Gradiente?**

        El descenso de gradiente es un método iterativo para encontrar el mínimo de una función.  
        En este ejemplo, queremos minimizar la función:

        $$
        f(x) = x^2
        $$

        cuyo mínimo está en \( x = 0 \).

        - Calculamos la derivada:  
        $$
        f'(x) = 2x
        $$
        - Partimos de un valor inicial \( x₀ \).
        - En cada paso, actualizamos \( x \) restando un pequeño valor proporcional a la derivada, controlado por la tasa de aprendizaje (*learning rate*).
        - El proceso se repite varias veces (epochs), y observamos cómo \( x \) se acerca al mínimo.

        Ajusta los parámetros para ver cómo afecta el proceso de optimización.
        """)

        lr = st.slider("Tasa de aprendizaje (learning rate)", 0.01, 1.0, 0.1)
        x0 = st.slider("Valor inicial x₀", -10.0, 10.0, 5.0)
        epochs = st.slider("Número de iteraciones (epochs)", 1, 200, 50)

        def grad_desc(x_init, lr, epochs):
            x = x_init
            hist = []
            for _ in range(epochs):
                grad = 2*x
                x = x - lr * grad
                hist.append(x)
            return hist

        hist = grad_desc(x0, lr, epochs)
        st.line_chart(hist)

    with img_tab:
        st.subheader("Compresión de imagen (SVD, escala de grises)")

        st.markdown(r"""
        **¿Por qué la compresión de imágenes con SVD es un método numérico?**
        La descomposición en valores singulares (SVD) es un algoritmo numérico que factoriza una matriz en tres matrices:""")
        st.latex(r"A = U \ Σ V^T")

        st.markdown("""donde:  
        - \(A\) es la matriz original (la imagen en escala de grises).  
        - \(U\) y \(V^T\) son matrices ortogonales.  
        - \(Σ\) es una matriz diagonal con valores singulares ordenados de mayor a menor.""")

        st.markdown("Al conservar solo las primeras \(k\) componentes singulares, obtenemos una **aproximación** de la imagen original:")

        st.latex(r"A_k = U_k \Sigma_k V_k^T")

        st.markdown(r"""Esto permite reducir la cantidad de datos almacenados y es un ejemplo claro de cómo los métodos numéricos permiten resolver problemas prácticos de gran tamaño mediante aproximaciones computacionales.
        Ajusta el número de componentes para ver cómo cambia la calidad y tamaño de la imagen comprimida.
        """)

        uploaded = st.file_uploader("Sube una imagen (jpg/png)", type=["jpg","jpeg","png"])
        if uploaded is not None:
            # Abrimos y convertimos a escala de grises
            img = Image.open(uploaded).convert("L")  
            arr = np.array(img).astype(float)

            max_k = min(arr.shape)
            k = st.slider("Componentes (k)", 1, max_k, min(50, max_k))

            # Descomposición SVD
            U, S, Vt = np.linalg.svd(arr, full_matrices=False)
            arr_approx = (U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :])
            arr_approx = np.clip(arr_approx, 0, 255).astype(np.uint8)

            col1, col2 = st.columns(2)
            col1.image(img, caption="Original", use_container_width=True)
            col2.image(arr_approx, caption=f"Comprimida k={k}", use_container_width=True)

            st.write("Reducción de datos aproximada:", f"{k * (arr.shape[0] + arr.shape[1])} elementos (vs {arr.size})")

            # --- Checkbox para diferencia ---
            show_diff = st.checkbox("Mostrar diferencia (error visual)")
            if show_diff:
                diff = np.abs(arr - arr_approx).astype(np.uint8)
                st.image(diff, caption="Diferencia (|Original - Comprimida|)", use_container_width=True)


    with fin_tab:
        st.subheader("Finanzas: Simulación Monte Carlo para Precios Futuros")

        st.markdown("""
        Esta sección simula posibles precios futuros de un activo financiero (por ejemplo, una acción) en **bolivianos (BOB)** después de un tiempo determinado.
        Usamos muchas simulaciones con números aleatorios para representar la incertidumbre del mercado y obtener una distribución probable de precios.

        Así podemos estimar no solo un valor, sino un rango de posibles resultados, lo que ayuda a entender y tomar decisiones en situaciones complejas donde no es posible calcular un resultado exacto.
        """)

        S0 = st.number_input("Precio inicial S0 (en BOB)", value=100)
        mu = st.number_input("Tasa esperada (mu, en decimal, ej. 0.05 para 5%)", value=0.05)
        sigma = st.number_input("Volatilidad (sigma, en decimal, ej. 0.2 para 20%)", value=0.2)
        T = st.number_input("Tiempo (años)", value=1)
        sims = st.slider("Número de simulaciones", 100, 20000, 5000)

        Z = np.random.normal(0, 1, sims)
        ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

        st.write(f"Precio esperado (media) después de {T} años: {np.mean(ST):.2f} BOB")

        fig = px.histogram(ST, nbins=50, title="Distribución de precios simulados (BOB)")
        st.plotly_chart(fig)

    with conclusion:
        st.header("Conclusión")
        st.markdown("""
        Los **métodos numéricos** son técnicas fundamentales para resolver problemas matemáticos que no pueden ser solucionados de forma exacta o analítica.  
        Se utilizan ampliamente en ingeniería, física, finanzas, inteligencia artificial, y muchas otras áreas para aproximar soluciones con alta precisión.

        Python es un lenguaje ideal para implementar métodos numéricos por su facilidad, potencia y gran ecosistema de bibliotecas (como NumPy, SciPy, Matplotlib, Pandas, etc.).  
        Esto permite desarrollar simulaciones, optimizaciones y análisis de datos de forma eficiente y con gran soporte comunitario.

        ### Recursos recomendados para profundizar
        - **Documentación oficial de Python**: [https://docs.python.org/3/](https://docs.python.org/3/)  
        - **Curso de Python para Ciencia de Datos y Métodos Numéricos (YouTube)**:  
        [https://www.youtube.com/watch?v=rfscVS0vtbw](https://www.youtube.com/watch?v=rfscVS0vtbw)  
        - **Documentación Streamlit** para crear apps interactivas: [https://docs.streamlit.io/](https://docs.streamlit.io/)  
        - **Libros y artículos sobre métodos numéricos** y aplicaciones prácticas en finanzas, física, IA, etc.
        """)

        st.subheader("Video tutorial recomendado")
        st.video("https://www.youtube.com/watch?v=WC4_YpdgE18&list=PL7HAy5R0ehQXnHqAEJUNb1ci_24cUQ4L5")



