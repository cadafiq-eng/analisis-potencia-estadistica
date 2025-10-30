import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, f, t
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Configuración de estilo
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')
    
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# ==================== FUNCIONES DE CÁLCULO ====================

def potencia_media(mu0, mu1, sigma, n, alpha, tipo='bilateral'):
    """Prueba Z para muestras grandes"""
    se = sigma / np.sqrt(n)
    if tipo == 'bilateral':
        z_alpha = norm.ppf(1 - alpha / 2)
        lim_inf = mu0 - z_alpha * se
        lim_sup = mu0 + z_alpha * se
        beta = norm.cdf((lim_sup - mu1) / se) - norm.cdf((lim_inf - mu1) / se)
    else:
        z_alpha = norm.ppf(1 - alpha)
        if mu1 > mu0:
            lim = mu0 + z_alpha * se
            beta = norm.cdf((lim - mu1) / se)
            lim_inf = lim
            lim_sup = None
        else:
            lim = mu0 - z_alpha * se
            beta = 1 - norm.cdf((lim - mu1) / se)
            lim_inf = lim
            lim_sup = None
    potencia = 1 - beta
    return beta, potencia, lim_inf, lim_sup

def potencia_t_student(mu0, mu1, sigma, n, alpha, tipo='bilateral'):
    """Prueba t para muestras pequeñas"""
    se = sigma / np.sqrt(n)
    df = n - 1
    
    if tipo == 'bilateral':
        t_alpha = t.ppf(1 - alpha / 2, df)
        lim_inf = mu0 - t_alpha * se
        lim_sup = mu0 + t_alpha * se
        
        # Distribución t no central
        delta = (mu1 - mu0) / se
        # Aproximación usando distribución normal
        beta = norm.cdf((lim_sup - mu1) / se) - norm.cdf((lim_inf - mu1) / se)
    else:
        t_alpha = t.ppf(1 - alpha, df)
        if mu1 > mu0:
            lim = mu0 + t_alpha * se
            delta = (mu1 - mu0) / se
            beta = norm.cdf((lim - mu1) / se)
            lim_inf = lim
            lim_sup = None
        else:
            lim = mu0 - t_alpha * se
            delta = (mu1 - mu0) / se
            beta = 1 - norm.cdf((lim - mu1) / se)
            lim_inf = lim
            lim_sup = None
    
    potencia = 1 - beta
    return beta, potencia, lim_inf, lim_sup, df

def potencia_diferencia_medias(mu1, mu2, sigma1, sigma2, n1, n2, alpha, tipo='bilateral'):
    """Prueba para diferencia de medias (dos muestras independientes)"""
    # Bajo H0: mu1 - mu2 = 0
    # Bajo H1: mu1 - mu2 = delta (donde delta = mu1 - mu2 observado)
    
    se_pooled = np.sqrt((sigma1**2 / n1) + (sigma2**2 / n2))
    delta_obs = mu1 - mu2  # Diferencia observada/esperada
    
    if tipo == 'bilateral':
        z_alpha = norm.ppf(1 - alpha / 2)
        lim_inf = 0 - z_alpha * se_pooled
        lim_sup = 0 + z_alpha * se_pooled
        
        # Beta bajo H1
        beta = norm.cdf((lim_sup - delta_obs) / se_pooled) - norm.cdf((lim_inf - delta_obs) / se_pooled)
    else:
        z_alpha = norm.ppf(1 - alpha)
        if delta_obs > 0:  # mu1 > mu2
            lim = 0 + z_alpha * se_pooled
            beta = norm.cdf((lim - delta_obs) / se_pooled)
            lim_inf = lim
            lim_sup = None
        else:  # mu1 < mu2
            lim = 0 - z_alpha * se_pooled
            beta = 1 - norm.cdf((lim - delta_obs) / se_pooled)
            lim_inf = lim
            lim_sup = None
    
    potencia = 1 - beta
    return beta, potencia, lim_inf, lim_sup, se_pooled

def potencia_diferencia_proporciones(p1, p2, n1, n2, alpha, tipo='bilateral'):
    """Prueba para diferencia de proporciones (dos grupos)"""
    # Bajo H0: p1 - p2 = 0
    # Bajo H1: p1 - p2 = delta
    
    # Proporción pooled bajo H0
    p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se_h0 = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    # Error estándar bajo H1
    se_h1 = np.sqrt((p1 * (1-p1) / n1) + (p2 * (1-p2) / n2))
    
    delta_obs = p1 - p2
    
    if tipo == 'bilateral':
        z_alpha = norm.ppf(1 - alpha / 2)
        lim_inf = 0 - z_alpha * se_h0
        lim_sup = 0 + z_alpha * se_h0
        
        beta = norm.cdf((lim_sup - delta_obs) / se_h1) - norm.cdf((lim_inf - delta_obs) / se_h1)
    else:
        z_alpha = norm.ppf(1 - alpha)
        if delta_obs > 0:
            lim = 0 + z_alpha * se_h0
            beta = norm.cdf((lim - delta_obs) / se_h1)
            lim_inf = lim
            lim_sup = None
        else:
            lim = 0 - z_alpha * se_h0
            beta = 1 - norm.cdf((lim - delta_obs) / se_h1)
            lim_inf = lim
            lim_sup = None
    
    potencia = 1 - beta
    return beta, potencia, lim_inf, lim_sup

def potencia_proporcion(p0, p1, n, alpha, tipo='bilateral'):
    """Prueba de una proporción"""
    se0 = np.sqrt(p0 * (1 - p0) / n)
    if tipo == 'bilateral':
        z_alpha = norm.ppf(1 - alpha / 2)
        lim_inf = p0 - z_alpha * se0
        lim_sup = p0 + z_alpha * se0
        se1 = np.sqrt(p1 * (1 - p1) / n)
        beta = norm.cdf((lim_sup - p1) / se1) - norm.cdf((lim_inf - p1) / se1)
    else:
        z_alpha = norm.ppf(1 - alpha)
        se1 = np.sqrt(p1 * (1 - p1) / n)
        if p1 > p0:
            lim = p0 + z_alpha * se0
            beta = norm.cdf((lim - p1) / se1)
            lim_inf = lim
            lim_sup = None
        else:
            lim = p0 - z_alpha * se0
            beta = 1 - norm.cdf((lim - p1) / se1)
            lim_inf = lim
            lim_sup = None
    potencia = 1 - beta
    return beta, potencia, lim_inf, lim_sup

def potencia_varianza(sigma0, sigma1, n, alpha, tipo='bilateral'):
    """Prueba de varianza"""
    df = n - 1
    if tipo == 'bilateral':
        chi2_sup = chi2.ppf(1 - alpha / 2, df)
        chi2_inf = chi2.ppf(alpha / 2, df)
        lim_inf = (df * sigma0**2) / chi2_sup
        lim_sup = (df * sigma0**2) / chi2_inf
        chi2_stat_inf = (df * sigma1**2) / lim_inf
        chi2_stat_sup = (df * sigma1**2) / lim_sup
        beta = chi2.cdf(chi2_stat_sup, df) - chi2.cdf(chi2_stat_inf, df)
    else:
        if sigma1 > sigma0:
            chi2_crit = chi2.ppf(1 - alpha, df)
            lim_inf = (df * sigma0**2) / chi2_crit
            lim_sup = None
            chi2_stat = (df * sigma1**2) / lim_inf
            beta = chi2.cdf(chi2_stat, df)
        else:
            chi2_crit = chi2.ppf(alpha, df)
            lim_sup = (df * sigma0**2) / chi2_crit
            lim_inf = None
            chi2_stat = (df * sigma1**2) / lim_sup
            beta = 1 - chi2.cdf(chi2_stat, df)
    potencia = 1 - beta
    return beta, potencia, lim_inf, lim_sup

def potencia_razon_varianzas(sigma1, sigma2, n1, n2, alpha):
    """Prueba F para razón de varianzas"""
    df1 = n1 - 1
    df2 = n2 - 1
    f_sup = f.ppf(1 - alpha / 2, df1, df2)
    f_inf = f.ppf(alpha / 2, df1, df2)
    r0 = 1
    r1 = (sigma1**2) / (sigma2**2)
    lim_inf = r0 / f_sup
    lim_sup = r0 / f_inf
    f_stat_inf = r1 / lim_inf
    f_stat_sup = r1 / lim_sup
    beta = f.cdf(f_stat_sup, df1, df2) - f.cdf(f_stat_inf, df1, df2)
    potencia = 1 - beta
    return beta, potencia, lim_inf, lim_sup

# ==================== FUNCIONES DE VISUALIZACIÓN ====================

def graficar_distribucion(mu0, mu1, se, lim_inf, lim_sup, tipo='bilateral', titulo=""):
    """Gráfico de distribuciones con regiones de error"""
    fig, ax = plt.subplots(figsize=(12, 7))
    rango = max(abs(mu0 - mu1), 4*se)
    x = np.linspace(min(mu0, mu1) - rango, max(mu0, mu1) + rango, 1000)
    y0 = norm.pdf(x, mu0, se)
    y1 = norm.pdf(x, mu1, se)
    ax.plot(x, y0, label='H₀', color='blue', linewidth=2)
    ax.plot(x, y1, label='H₁', color='red', linewidth=2)
    if tipo == 'bilateral':
        ax.axvline(lim_inf, color='black', linestyle='--', linewidth=1.5)
        ax.axvline(lim_sup, color='black', linestyle='--', linewidth=1.5)
        ax.fill_between(x, 0, y0, where=(x < lim_inf) | (x > lim_sup), color='blue', alpha=0.3, label='α')
        ax.fill_between(x, 0, y1, where=(x >= lim_inf) & (x <= lim_sup), color='orange', alpha=0.4, label='β')
        ax.fill_between(x, 0, y1, where=(x < lim_inf) | (x > lim_sup), color='green', alpha=0.3, label='Potencia')
    else:
        ax.axvline(lim_inf, color='black', linestyle='--', linewidth=1.5)
        if mu1 > mu0:
            ax.fill_between(x, 0, y0, where=(x > lim_inf), color='blue', alpha=0.3, label='α')
            ax.fill_between(x, 0, y1, where=(x <= lim_inf), color='orange', alpha=0.4, label='β')
            ax.fill_between(x, 0, y1, where=(x > lim_inf), color='green', alpha=0.3, label='Potencia')
        else:
            ax.fill_between(x, 0, y0, where=(x < lim_inf), color='blue', alpha=0.3, label='α')
            ax.fill_between(x, 0, y1, where=(x >= lim_inf), color='orange', alpha=0.4, label='β')
            ax.fill_between(x, 0, y1, where=(x < lim_inf), color='green', alpha=0.3, label='Potencia')
    ax.set_xlabel('Valor del estadístico', fontsize=12)
    ax.set_ylabel('Densidad', fontsize=12)
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.close()
    return fig

def exportar_excel(dataframe):
    """Exporta DataFrame a Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        dataframe.to_excel(writer, index=False, sheet_name='Resultados')
    return output.getvalue()

def exportar_pdf(texto):
    """Exporta resultados a PDF"""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Analisis de Potencia Estadistica")
    c.setFont("Helvetica", 12)
    y_position = height - 100
    for line in texto.split('\n'):
        if y_position < 50:
            c.showPage()
            y_position = height - 50
        c.drawString(50, y_position, line)
        y_position -= 20
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# ==================== INTERFAZ STREAMLIT ====================

st.set_page_config(page_title="Analisis de Potencia", layout="wide", page_icon="📊")

with st.sidebar:
    st.title("📚 Guia Rapida")
    st.markdown("""
    ### Conceptos Clave
    **Error Tipo I (α):** Rechazar H₀ cuando es verdadera
    
    **Error Tipo II (β):** No rechazar H₀ cuando es falsa
    
    **Potencia (1-β):** Rechazar H₀ cuando es falsa (ideal ≥ 0.80)
    
    ---
    
    ### Tipos de Pruebas
    - **Z-test:** n ≥ 30, σ conocida
    - **t-test:** n < 30, σ desconocida
    - **Diferencias:** Comparar 2 grupos
    """)

st.title("🔍 Analisis de Potencia y Error Tipo II")
st.markdown("Explora como diferentes factores afectan la potencia estadistica de tus pruebas de hipotesis.")

tabs = st.tabs([
    "📐 Media (Z)", 
    "📏 Media (t-Student)",
    "⚖️ Diferencia de Medias",
    "📊 Proporcion",
    "📈 Diferencia de Proporciones",
    "📉 Varianza", 
    "🔄 Razon de Varianzas"
])

# ==================== TAB 1: MEDIA (Z) ====================
with tabs[0]:
    st.header("Prueba Z sobre la Media (Muestras Grandes)")
    with st.expander("ℹ️ Informacion"):
        st.markdown("""
        **Cuándo usar:** Cuando n ≥ 30 y se conoce σ poblacional
        
        **Supuestos:**
        - Muestra aleatoria
        - Población normal o n grande (TCL)
        - Varianza poblacional conocida
        
        **Ejemplo:** Verificar si el tiempo promedio de espera es diferente de 10 minutos.
        """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Parametros")
        mu0_z = st.slider("Media bajo H₀ (μ₀)", 50.0, 150.0, 100.0, 0.1, key="mu0_z")
        mu1_z = st.slider("Media verdadera (μ₁)", 50.0, 150.0, 105.0, 0.1, key="mu1_z")
        sigma_z = st.slider("Desviacion estandar (σ)", 1.0, 50.0, 15.0, 0.1, key="sigma_z")
        n_z = st.slider("Tamano de muestra (n)", 30, 500, 100, 1, key="n_z")
        alpha_z = st.select_slider("Nivel de significancia (α)", options=[0.01, 0.05, 0.10], value=0.05, key="alpha_z")
        tipo_z = st.radio("Tipo de prueba", ["bilateral", "unilateral"], key="tipo_z")
    
    with col2:
        st.subheader("Resultados")
        beta_z, potencia_z, lim_inf_z, lim_sup_z = potencia_media(mu0_z, mu1_z, sigma_z, n_z, alpha_z, tipo_z)
        st.metric("Error Tipo II (β)", f"{beta_z:.4f}")
        st.metric("Potencia (1-β)", f"{potencia_z:.4f}")
        if potencia_z >= 0.80:
            st.success("✅ Potencia adecuada")
        elif potencia_z >= 0.60:
            st.warning("⚠️ Potencia moderada")
        else:
            st.error("❌ Potencia baja")
        st.info(f"Tamano del efecto: d = {abs(mu1_z-mu0_z)/sigma_z:.3f}")
    
    st.subheader("Visualizacion")
    fig_z = graficar_distribucion(mu0_z, mu1_z, sigma_z/np.sqrt(n_z), lim_inf_z, lim_sup_z, tipo_z, "Distribuciones Z")
    st.pyplot(fig_z)
    
    df_z = pd.DataFrame([{"μ₀": mu0_z, "μ₁": mu1_z, "σ": sigma_z, "n": n_z, "α": alpha_z, "β": round(beta_z, 4), "Potencia": round(potencia_z, 4)}])
    st.dataframe(df_z, use_container_width=True)
    st.download_button("📥 Excel", data=exportar_excel(df_z), file_name="potencia_media_z.xlsx", key="dl_z")

# ==================== TAB 2: MEDIA (t-Student) ====================
with tabs[1]:
    st.header("Prueba t sobre la Media (Muestras Pequeñas)")
    with st.expander("ℹ️ Informacion"):
        st.markdown("""
        **Cuándo usar:** Cuando n < 30 y/o σ desconocida
        
        **Supuestos:**
        - Muestra aleatoria
        - Población normal (importante para n pequeña)
        - Varianza poblacional desconocida (se estima con s)
        
        **Ejemplo:** Probar si el promedio de calificaciones de un grupo pequeño de estudiantes difiere de 75.
        """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Parametros")
        mu0_t = st.slider("Media bajo H₀ (μ₀)", 50.0, 150.0, 100.0, 0.1, key="mu0_t")
        mu1_t = st.slider("Media verdadera (μ₁)", 50.0, 150.0, 105.0, 0.1, key="mu1_t")
        sigma_t = st.slider("Desviacion estandar (σ o s)", 1.0, 50.0, 15.0, 0.1, key="sigma_t")
        n_t = st.slider("Tamano de muestra (n)", 5, 50, 20, 1, key="n_t")
        alpha_t = st.select_slider("Nivel de significancia (α)", options=[0.01, 0.05, 0.10], value=0.05, key="alpha_t")
        tipo_t = st.radio("Tipo de prueba", ["bilateral", "unilateral"], key="tipo_t")
    
    with col2:
        st.subheader("Resultados")
        beta_t, potencia_t, lim_inf_t, lim_sup_t, df_t = potencia_t_student(mu0_t, mu1_t, sigma_t, n_t, alpha_t, tipo_t)
        st.metric("Error Tipo II (β)", f"{beta_t:.4f}")
        st.metric("Potencia (1-β)", f"{potencia_t:.4f}")
        if potencia_t >= 0.80:
            st.success("✅ Potencia adecuada")
        elif potencia_t >= 0.60:
            st.warning("⚠️ Potencia moderada")
        else:
            st.error("❌ Potencia baja")
        st.info(f"""
        **Grados de libertad:** {df_t}
        
        **Tamano del efecto:** d = {abs(mu1_t-mu0_t)/sigma_t:.3f}
        """)
    
    st.subheader("Visualizacion")
    fig_t = graficar_distribucion(mu0_t, mu1_t, sigma_t/np.sqrt(n_t), lim_inf_t, lim_sup_t, tipo_t, "Distribuciones t-Student")
    st.pyplot(fig_t)
    
    df_result_t = pd.DataFrame([{"μ₀": mu0_t, "μ₁": mu1_t, "σ": sigma_t, "n": n_t, "gl": df_t, "α": alpha_t, "β": round(beta_t, 4), "Potencia": round(potencia_t, 4)}])
    st.dataframe(df_result_t, use_container_width=True)
    st.download_button("📥 Excel", data=exportar_excel(df_result_t), file_name="potencia_t_student.xlsx", key="dl_t")

# ==================== TAB 3: DIFERENCIA DE MEDIAS ====================
with tabs[2]:
    st.header("Prueba para Diferencia de Medias (Dos Muestras)")
    with st.expander("ℹ️ Informacion"):
        st.markdown("""
        **Cuándo usar:** Comparar las medias de dos grupos independientes
        
        **Hipótesis:**
        - H₀: μ₁ = μ₂ (o μ₁ - μ₂ = 0)
        - H₁: μ₁ ≠ μ₂ (bilateral) o μ₁ > μ₂ o μ₁ < μ₂ (unilateral)
        
        **Ejemplo:** ¿El grupo de tratamiento tiene un promedio diferente al grupo control?
        """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Parametros")
        st.markdown("**Grupo 1:**")
        mu1_diff = st.slider("Media grupo 1 (μ₁)", 50.0, 150.0, 105.0, 0.1, key="mu1_diff")
        sigma1_diff = st.slider("Desv. est. grupo 1 (σ₁)", 1.0, 50.0, 15.0, 0.1, key="sigma1_diff")
        n1_diff = st.slider("Tamano muestra grupo 1 (n₁)", 5, 500, 50, 1, key="n1_diff")
        
        st.markdown("**Grupo 2:**")
        mu2_diff = st.slider("Media grupo 2 (μ₂)", 50.0, 150.0, 100.0, 0.1, key="mu2_diff")
        sigma2_diff = st.slider("Desv. est. grupo 2 (σ₂)", 1.0, 50.0, 15.0, 0.1, key="sigma2_diff")
        n2_diff = st.slider("Tamano muestra grupo 2 (n₂)", 5, 500, 50, 1, key="n2_diff")
        
        alpha_diff = st.select_slider("Nivel de significancia (α)", options=[0.01, 0.05, 0.10], value=0.05, key="alpha_diff")
        tipo_diff = st.radio("Tipo de prueba", ["bilateral", "unilateral"], key="tipo_diff")
    
    with col2:
        st.subheader("Resultados")
        beta_diff, potencia_diff, lim_inf_diff, lim_sup_diff, se_diff = potencia_diferencia_medias(
            mu1_diff, mu2_diff, sigma1_diff, sigma2_diff, n1_diff, n2_diff, alpha_diff, tipo_diff)
        
        st.metric("Diferencia (μ₁ - μ₂)", f"{mu1_diff - mu2_diff:.2f}")
        st.metric("Error Tipo II (β)", f"{beta_diff:.4f}")
        st.metric("Potencia (1-β)", f"{potencia_diff:.4f}")
        
        if potencia_diff >= 0.80:
            st.success("✅ Potencia adecuada")
        else:
            st.warning("⚠️ Aumentar tamaños de muestra")
        
        # Cohen's d para dos muestras
        s_pooled = np.sqrt(((n1_diff-1)*sigma1_diff**2 + (n2_diff-1)*sigma2_diff**2) / (n1_diff+n2_diff-2))
        cohens_d = abs(mu1_diff - mu2_diff) / s_pooled
        st.info(f"Tamano del efecto (d): {cohens_d:.3f}")
    
    st.subheader("Visualizacion")
    fig_diff = graficar_distribucion(0, mu1_diff - mu2_diff, se_diff, lim_inf_diff, lim_sup_diff, tipo_diff, 
                                     "Distribucion de la Diferencia de Medias")
    st.pyplot(fig_diff)
    
    df_diff_result = pd.DataFrame([{
        "μ₁": mu1_diff, "σ₁": sigma1_diff, "n₁": n1_diff,
        "μ₂": mu2_diff, "σ₂": sigma2_diff, "n₂": n2_diff,
        "Diferencia": round(mu1_diff - mu2_diff, 2),
        "α": alpha_diff, "β": round(beta_diff, 4), "Potencia": round(potencia_diff, 4)
    }])
    st.dataframe(df_diff_result, use_container_width=True)
    st.download_button("📥 Excel", data=exportar_excel(df_diff_result), file_name="potencia_dif_medias.xlsx", key="dl_diff")

# ==================== TAB 4: PROPORCION ====================
with tabs[3]:
    st.header("Prueba sobre una Proporcion")
    with st.expander("ℹ️ Informacion"):
        st.markdown("""
        **Cuándo usar:** Comparar una proporción con un valor hipotético
        
        **Ejemplo:** ¿La tasa de conversión es diferente del 50%?
        """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Parametros")
        p0 = st.slider("Proporcion bajo H₀ (p₀)", 0.0, 1.0, 0.5, 0.01, key="p0")
        p1 = st.slider("Proporcion verdadera (p₁)", 0.0, 1.0, 0.6, 0.01, key="p1")
        n_prop = st.slider("Tamano de muestra (n)", 10, 1000, 100, 1, key="n_prop")
        alpha_prop = st.select_slider("Nivel de significancia (α)", options=[0.01, 0.05, 0.10], value=0.05, key="alpha_prop")
        tipo_prop = st.radio("Tipo de prueba", ["bilateral", "unilateral"], key="tipo_prop")
    
    with col2:
        st.subheader("Resultados")
        beta_prop, potencia_prop, lim_inf_prop, lim_sup_prop = potencia_proporcion(p0, p1, n_prop, alpha_prop, tipo_prop)
        st.metric("Error Tipo II (β)", f"{beta_prop:.4f}")
        st.metric("Potencia (1-β)", f"{potencia_prop:.4f}")
        if potencia_prop >= 0.80:
            st.success("✅ Potencia adecuada")
        else:
            st.warning("⚠️ Aumentar n")
        st.info(f"Tamano del efecto: h = {abs(2*np.arcsin(np.sqrt(p1)) - 2*np.arcsin(np.sqrt(p0))):.3f}")
    
    st.subheader("Visualizacion")
    se_prop = np.sqrt(p0*(1-p0)/n_prop)
    fig_prop = graficar_distribucion(p0, p1, se_prop, lim_inf_prop, lim_sup_prop, tipo_prop, "Distribuciones de Proporcion")
    st.pyplot(fig_prop)
    
    df_prop_result = pd.DataFrame([{"p₀": p0, "p₁": p1, "n": n_prop, "α": alpha_prop, "β": round(beta_prop, 4), "Potencia": round(potencia_prop, 4)}])
    st.dataframe(df_prop_result, use_container_width=True)
    st.download_button("📥 Excel", data=exportar_excel(df_prop_result), file_name="potencia_proporcion.xlsx", key="dl_prop")

# ==================== TAB 5: DIFERENCIA DE PROPORCIONES ====================
with tabs[4]:
    st.header("Prueba para Diferencia de Proporciones (Dos Grupos)")
    with st.expander("ℹ️ Informacion"):
        st.markdown("""
        **Cuándo usar:** Comparar las proporciones de éxito entre dos grupos independientes
        
        **Hipótesis:**
        - H₀: p₁ = p₂ (o p₁ - p₂ = 0)
        - H₁: p₁ ≠ p₂ (bilateral) o p₁ > p₂ o p₁ < p₂ (unilateral)
        
        **Ejemplo:** 
        - ¿La tasa de conversión del grupo A es diferente a la del grupo B?
        - ¿El tratamiento mejora la tasa de recuperación comparado con el placebo?
        """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Parametros")
        st.markdown("**Grupo 1:**")
        p1_diff_prop = st.slider("Proporcion grupo 1 (p₁)", 0.0, 1.0, 0.6, 0.01, key="p1_diff_prop")
        n1_diff_prop = st.slider("Tamano muestra grupo 1 (n₁)", 10, 1000, 100, 1, key="n1_diff_prop")
        
        st.markdown("**Grupo 2:**")
        p2_diff_prop = st.slider("Proporcion grupo 2 (p₂)", 0.0, 1.0, 0.5, 0.01, key="p2_diff_prop")
        n2_diff_prop = st.slider("Tamano muestra grupo 2 (n₂)", 10, 1000, 100, 1, key="n2_diff_prop")
        
        alpha_diff_prop = st.select_slider("Nivel de significancia (α)", options=[0.01, 0.05, 0.10], value=0.05, key="alpha_diff_prop")
        tipo_diff_prop = st.radio("Tipo de prueba", ["bilateral", "unilateral"], key="tipo_diff_prop")
    
    with col2:
        st.subheader("Resultados")
        beta_diff_prop, potencia_diff_prop, lim_inf_diff_prop, lim_sup_diff_prop = potencia_diferencia_proporciones(
            p1_diff_prop, p2_diff_prop, n1_diff_prop, n2_diff_prop, alpha_diff_prop, tipo_diff_prop)
        
        diferencia_prop = p1_diff_prop - p2_diff_prop
        st.metric("Diferencia (p₁ - p₂)", f"{diferencia_prop:.4f}")
        st.metric("Error Tipo II (β)", f"{beta_diff_prop:.4f}")
        st.metric("Potencia (1-β)", f"{potencia_diff_prop:.4f}")
        
        if potencia_diff_prop >= 0.80:
            st.success("✅ Potencia adecuada")
        else:
            st.warning("⚠️ Aumentar tamaños de muestra")
        
        # h de Cohen para diferencia de proporciones
        h_cohen = 2 * (np.arcsin(np.sqrt(p1_diff_prop)) - np.arcsin(np.sqrt(p2_diff_prop)))
        st.info(f"Tamano del efecto (h): {abs(h_cohen):.3f}")
    
    st.subheader("Visualizacion")
    p_pooled = (p1_diff_prop * n1_diff_prop + p2_diff_prop * n2_diff_prop) / (n1_diff_prop + n2_diff_prop)
    se_diff_prop = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1_diff_prop + 1/n2_diff_prop))
    fig_diff_prop = graficar_distribucion(0, diferencia_prop, se_diff_prop, lim_inf_diff_prop, lim_sup_diff_prop, 
                                          tipo_diff_prop, "Distribucion de la Diferencia de Proporciones")
    st.pyplot(fig_diff_prop)
    
    df_diff_prop_result = pd.DataFrame([{
        "p₁": p1_diff_prop, "n₁": n1_diff_prop,
        "p₂": p2_diff_prop, "n₂": n2_diff_prop,
        "Diferencia": round(diferencia_prop, 4),
        "α": alpha_diff_prop, "β": round(beta_diff_prop, 4), "Potencia": round(potencia_diff_prop, 4)
    }])
    st.dataframe(df_diff_prop_result, use_container_width=True)
    st.download_button("📥 Excel", data=exportar_excel(df_diff_prop_result), file_name="potencia_dif_proporciones.xlsx", key="dl_diff_prop")

# ==================== TAB 6: VARIANZA ====================
with tabs[5]:
    st.header("Prueba sobre la Varianza")
    with st.expander("ℹ️ Informacion"):
        st.markdown("Prueba si la varianza poblacional difiere de un valor especifico usando Chi-cuadrado.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Parametros")
        sigma0_var = st.slider("Desv. estandar bajo H₀ (σ₀)", 1.0, 50.0, 15.0, 0.1, key="s0v")
        sigma1_var = st.slider("Desv. estandar verdadera (σ₁)", 1.0, 50.0, 20.0, 0.1, key="s1v")
        n_var = st.slider("Tamano de muestra (n)", 5, 300, 36, 1, key="nv")
        alpha_var = st.select_slider("Nivel de significancia (α)", options=[0.01, 0.05, 0.10], value=0.05, key="av")
        tipo_var = st.radio("Tipo de prueba", ["bilateral", "unilateral"], key="tv")
    
    with col2:
        st.subheader("Resultados")
        beta_var, potencia_var, lim_inf_var, lim_sup_var = potencia_varianza(sigma0_var, sigma1_var, n_var, alpha_var, tipo_var)
        st.metric("Error Tipo II (β)", f"{beta_var:.4f}")
        st.metric("Potencia (1-β)", f"{potencia_var:.4f}")
        if potencia_var >= 0.80:
            st.success("✅ Potencia adecuada")
        else:
            st.warning("⚠️ Potencia insuficiente")
        st.info(f"Razon de varianzas: {(sigma1_var/sigma0_var)**2:.3f}")
    
    df_var_result = pd.DataFrame([{"σ₀": sigma0_var, "σ₁": sigma1_var, "n": n_var, "α": alpha_var, "β": round(beta_var, 4), "Potencia": round(potencia_var, 4)}])
    st.dataframe(df_var_result, use_container_width=True)
    st.download_button("📥 Excel", data=exportar_excel(df_var_result), file_name="potencia_varianza.xlsx", key="dl_var")

# ==================== TAB 7: RAZON DE VARIANZAS ====================
with tabs[6]:
    st.header("Prueba F: Razon entre Varianzas")
    with st.expander("ℹ️ Informacion"):
        st.markdown("Compara las varianzas de dos poblaciones independientes.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Parametros")
        sigma1_f = st.slider("Desv. est. grupo 1 (σ₁)", 1.0, 50.0, 15.0, 0.1, key="s1f")
        sigma2_f = st.slider("Desv. est. grupo 2 (σ₂)", 1.0, 50.0, 10.0, 0.1, key="s2f")
        n1_f = st.slider("Tamano muestra grupo 1 (n₁)", 5, 300, 36, 1, key="n1f")
        n2_f = st.slider("Tamano muestra grupo 2 (n₂)", 5, 300, 36, 1, key="n2f")
        alpha_f = st.select_slider("Nivel de significancia (α)", options=[0.01, 0.05, 0.10], value=0.05, key="af")
    
    with col2:
        st.subheader("Resultados")
        beta_f, potencia_f, lim_inf_f, lim_sup_f = potencia_razon_varianzas(sigma1_f, sigma2_f, n1_f, n2_f, alpha_f)
        st.metric("Error Tipo II (β)", f"{beta_f:.4f}")
        st.metric("Potencia (1-β)", f"{potencia_f:.4f}")
        if potencia_f >= 0.80:
            st.success("✅ Potencia adecuada")
        else:
            st.warning("⚠️ Potencia insuficiente")
        st.info(f"F = {(sigma1_f/sigma2_f)**2:.3f}")
    
    df_f_result = pd.DataFrame([{"σ₁": sigma1_f, "σ₂": sigma2_f, "n₁": n1_f, "n₂": n2_f, "α": alpha_f, "β": round(beta_f, 4), "Potencia": round(potencia_f, 4)}])
    st.dataframe(df_f_result, use_container_width=True)
    st.download_button("📥 Excel", data=exportar_excel(df_f_result), file_name="potencia_razon_varianzas.xlsx", key="dl_f")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown('<div style="text-align: center; color: gray;"><p>📊 Analisis de Potencia Estadistica | Streamlit</p><p><small>Herramienta educativa desarrollada para el aprendizaje de estadistica</small></p></div>', unsafe_allow_html=True)
