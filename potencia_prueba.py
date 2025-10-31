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

# Configuración
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

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
        delta = (mu1 - mu0) / se
        beta = norm.cdf((lim_sup - mu1) / se) - norm.cdf((lim_inf - mu1) / se)
    else:
        t_alpha = t.ppf(1 - alpha, df)
        if mu1 > mu0:
            lim = mu0 + t_alpha * se
            beta = norm.cdf((lim - mu1) / se)
            lim_inf = lim
            lim_sup = None
        else:
            lim = mu0 - t_alpha * se
            beta = 1 - norm.cdf((lim - mu1) / se)
            lim_inf = lim
            lim_sup = None
    potencia = 1 - beta
    return beta, potencia, lim_inf, lim_sup, df

def potencia_diferencia_medias(mu1, mu2, sigma1, sigma2, n1, n2, alpha, tipo='bilateral'):
    """Prueba para diferencia de medias"""
    se_pooled = np.sqrt((sigma1**2 / n1) + (sigma2**2 / n2))
    delta_obs = mu1 - mu2
    if tipo == 'bilateral':
        z_alpha = norm.ppf(1 - alpha / 2)
        lim_inf = 0 - z_alpha * se_pooled
        lim_sup = 0 + z_alpha * se_pooled
        beta = norm.cdf((lim_sup - delta_obs) / se_pooled) - norm.cdf((lim_inf - delta_obs) / se_pooled)
    else:
        z_alpha = norm.ppf(1 - alpha)
        if delta_obs > 0:
            lim = 0 + z_alpha * se_pooled
            beta = norm.cdf((lim - delta_obs) / se_pooled)
            lim_inf = lim
            lim_sup = None
        else:
            lim = 0 - z_alpha * se_pooled
            beta = 1 - norm.cdf((lim - delta_obs) / se_pooled)
            lim_inf = lim
            lim_sup = None
    potencia = 1 - beta
    return beta, potencia, lim_inf, lim_sup, se_pooled

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

def potencia_diferencia_proporciones(p1, p2, n1, n2, alpha, tipo='bilateral'):
    """Prueba para diferencia de proporciones"""
    p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se_h0 = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
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
    ax.set_xlabel('Valor', fontsize=12)
    ax.set_ylabel('Densidad', fontsize=12)
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.close()
    return fig

def exportar_excel(dataframe):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        dataframe.to_excel(writer, index=False, sheet_name='Resultados')
    return output.getvalue()

def exportar_pdf(texto):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Analisis de Potencia")
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

# ==================== INTERFAZ ====================

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
st.markdown("Herramienta para analisis de potencia estadistica en pruebas de hipotesis.")

tabs = st.tabs([
    "📐 Media (Z)",
    "📏 Media (t)",
    "⚖️ Dif. Medias",
    "📊 Proporcion",
    "📈 Dif. Proporciones",
    "📉 Varianza",
    "🔄 Razon Varianzas"
])

# ==================== TAB 1: MEDIA Z ====================
with tabs[0]:
    st.header("Prueba Z sobre la Media (n ≥ 30)")
    with st.expander("ℹ️ Informacion"):
        st.markdown("**Cuando usar:** n ≥ 30 y σ conocida. Ejemplo: Verificar si el tiempo promedio es diferente de 10 min.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        mu0_z = st.slider("Media H₀ (μ₀)", 50.0, 150.0, 100.0, 0.1, key="mu0z")
        mu1_z = st.slider("Media real (μ₁)", 50.0, 150.0, 105.0, 0.1, key="mu1z")
        sigma_z = st.slider("Desv. est. (σ)", 1.0, 50.0, 15.0, 0.1, key="sigmaz")
        n_z = st.slider("Muestra (n)", 30, 500, 100, 1, key="nz")
        alpha_z = st.select_slider("Significancia (α)", [0.01, 0.05, 0.10], value=0.05, key="alphaz")
        tipo_z = st.radio("Tipo", ["bilateral", "unilateral"], key="tipoz")
    
    with col2:
        beta_z, pot_z, lim_inf_z, lim_sup_z = potencia_media(mu0_z, mu1_z, sigma_z, n_z, alpha_z, tipo_z)
        st.metric("Error II (β)", f"{beta_z:.4f}")
        st.metric("Potencia", f"{pot_z:.4f}")
        if pot_z >= 0.80:
            st.success("✅ Adecuada")
        else:
            st.warning("⚠️ Baja")
        st.info(f"d = {abs(mu1_z-mu0_z)/sigma_z:.3f}")
    
    fig_z = graficar_distribucion(mu0_z, mu1_z, sigma_z/np.sqrt(n_z), lim_inf_z, lim_sup_z, tipo_z, "Prueba Z")
    st.pyplot(fig_z)
    
    df_z = pd.DataFrame([{"μ₀": mu0_z, "μ₁": mu1_z, "σ": sigma_z, "n": n_z, "α": alpha_z, "β": round(beta_z, 4), "Potencia": round(pot_z, 4)}])
    st.dataframe(df_z, use_container_width=True)
    st.download_button("📥 Excel", exportar_excel(df_z), "media_z.xlsx", key="dlz")

# ==================== TAB 2: MEDIA T ====================
with tabs[1]:
    st.header("Prueba t sobre la Media (n < 30)")
    with st.expander("ℹ️ Informacion"):
        st.markdown("**Cuando usar:** n < 30 o σ desconocida. Ejemplo: Probar si calificaciones difieren de 75.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        mu0_t = st.slider("Media H₀ (μ₀)", 50.0, 150.0, 100.0, 0.1, key="mu0t")
        mu1_t = st.slider("Media real (μ₁)", 50.0, 150.0, 105.0, 0.1, key="mu1t")
        sigma_t = st.slider("Desv. est. (s)", 1.0, 50.0, 15.0, 0.1, key="sigmat")
        n_t = st.slider("Muestra (n)", 5, 50, 20, 1, key="nt")
        alpha_t = st.select_slider("Significancia (α)", [0.01, 0.05, 0.10], value=0.05, key="alphat")
        tipo_t = st.radio("Tipo", ["bilateral", "unilateral"], key="tipot")
    
    with col2:
        beta_t, pot_t, lim_inf_t, lim_sup_t, df_t = potencia_t_student(mu0_t, mu1_t, sigma_t, n_t, alpha_t, tipo_t)
        st.metric("Error II (β)", f"{beta_t:.4f}")
        st.metric("Potencia", f"{pot_t:.4f}")
        if pot_t >= 0.80:
            st.success("✅ Adecuada")
        else:
            st.warning("⚠️ Baja")
        st.info(f"gl = {df_t}\nd = {abs(mu1_t-mu0_t)/sigma_t:.3f}")
    
    fig_t = graficar_distribucion(mu0_t, mu1_t, sigma_t/np.sqrt(n_t), lim_inf_t, lim_sup_t, tipo_t, "Prueba t")
    st.pyplot(fig_t)
    
    df_t_res = pd.DataFrame([{"μ₀": mu0_t, "μ₁": mu1_t, "s": sigma_t, "n": n_t, "gl": df_t, "α": alpha_t, "β": round(beta_t, 4), "Potencia": round(pot_t, 4)}])
    st.dataframe(df_t_res, use_container_width=True)
    st.download_button("📥 Excel", exportar_excel(df_t_res), "media_t.xlsx", key="dlt")

# ==================== TAB 3: DIFERENCIA MEDIAS ====================
with tabs[2]:
    st.header("Diferencia de Medias (2 grupos)")
    with st.expander("ℹ️ Informacion"):
        st.markdown("**Cuando usar:** Comparar medias de 2 grupos independientes. Ejemplo: Tratamiento vs. Control.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Grupo 1:**")
        mu1_dm = st.slider("μ₁", 50.0, 150.0, 105.0, 0.1, key="mu1dm")
        s1_dm = st.slider("σ₁", 1.0, 50.0, 15.0, 0.1, key="s1dm")
        n1_dm = st.slider("n₁", 5, 500, 50, 1, key="n1dm")
        st.markdown("**Grupo 2:**")
        mu2_dm = st.slider("μ₂", 50.0, 150.0, 100.0, 0.1, key="mu2dm")
        s2_dm = st.slider("σ₂", 1.0, 50.0, 15.0, 0.1, key="s2dm")
        n2_dm = st.slider("n₂", 5, 500, 50, 1, key="n2dm")
        alpha_dm = st.select_slider("Significancia (α)", [0.01, 0.05, 0.10], value=0.05, key="alphadm")
        tipo_dm = st.radio("Tipo", ["bilateral", "unilateral"], key="tipodm")
    
    with col2:
        beta_dm, pot_dm, lim_inf_dm, lim_sup_dm, se_dm = potencia_diferencia_medias(mu1_dm, mu2_dm, s1_dm, s2_dm, n1_dm, n2_dm, alpha_dm, tipo_dm)
        st.metric("Diferencia", f"{mu1_dm - mu2_dm:.2f}")
        st.metric("Error II (β)", f"{beta_dm:.4f}")
        st.metric("Potencia", f"{pot_dm:.4f}")
        if pot_dm >= 0.80:
            st.success("✅ Adecuada")
        else:
            st.warning("⚠️ Aumentar n")
        s_pool = np.sqrt(((n1_dm-1)*s1_dm**2 + (n2_dm-1)*s2_dm**2) / (n1_dm+n2_dm-2))
        st.info(f"d = {abs(mu1_dm - mu2_dm) / s_pool:.3f}")
    
    fig_dm = graficar_distribucion(0, mu1_dm - mu2_dm, se_dm, lim_inf_dm, lim_sup_dm, tipo_dm, "Diferencia de Medias")
    st.pyplot(fig_dm)
    
    df_dm_res = pd.DataFrame([{"μ₁": mu1_dm, "σ₁": s1_dm, "n₁": n1_dm, "μ₂": mu2_dm, "σ₂": s2_dm, "n₂": n2_dm, "Dif": round(mu1_dm - mu2_dm, 2), "α": alpha_dm, "β": round(beta_dm, 4), "Pot": round(pot_dm, 4)}])
    st.dataframe(df_dm_res, use_container_width=True)
    st.download_button("📥 Excel", exportar_excel(df_dm_res), "dif_medias.xlsx", key="dldm")

# ==================== TAB 4: PROPORCION ====================
with tabs[3]:
    st.header("Prueba sobre Proporcion")
    with st.expander("ℹ️ Informacion"):
        st.markdown("**Cuando usar:** Comparar proporcion con valor hipotetico. Ejemplo: Tasa de conversion diferente del 50%.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        p0_prop = st.slider("Proporcion H₀ (p₀)", 0.0, 1.0, 0.5, 0.01, key="p0prop")
        p1_prop = st.slider("Proporcion real (p₁)", 0.0, 1.0, 0.6, 0.01, key="p1prop")
        n_prop = st.slider("Muestra (n)", 10, 1000, 100, 1, key="nprop")
        alpha_prop = st.select_slider("Significancia (α)", [0.01, 0.05, 0.10], value=0.05, key="alphaprop")
        tipo_prop = st.radio("Tipo", ["bilateral", "unilateral"], key="tipoprop")
    
    with col2:
        beta_prop, pot_prop, lim_inf_prop, lim_sup_prop = potencia_proporcion(p0_prop, p1_prop, n_prop, alpha_prop, tipo_prop)
        st.metric("Error II (β)", f"{beta_prop:.4f}")
        st.metric("Potencia", f"{pot_prop:.4f}")
        if pot_prop >= 0.80:
            st.success("✅ Adecuada")
        else:
            st.warning("⚠️ Aumentar n")
        st.info(f"h = {abs(2*np.arcsin(np.sqrt(p1_prop)) - 2*np.arcsin(np.sqrt(p0_prop))):.3f}")
    
    se_prop = np.sqrt(p0_prop*(1-p0_prop)/n_prop)
    fig_prop = graficar_distribucion(p0_prop, p1_prop, se_prop, lim_inf_prop, lim_sup_prop, tipo_prop, "Proporcion")
    st.pyplot(fig_prop)
    
    df_prop_res = pd.DataFrame([{"p₀": p0_prop, "p₁": p1_prop, "n": n_prop, "α": alpha_prop, "β": round(beta_prop, 4), "Potencia": round(pot_prop, 4)}])
    st.dataframe(df_prop_res, use_container_width=True)
    st.download_button("📥 Excel", exportar_excel(df_prop_res), "proporcion.xlsx", key="dlprop")

# ==================== TAB 5: DIFERENCIA PROPORCIONES ====================
with tabs[4]:
    st.header("Diferencia de Proporciones (2 grupos)")
    with st.expander("ℹ️ Informacion"):
        st.markdown("**Cuando usar:** Comparar tasas de exito entre 2 grupos. Ejemplo: Tasa conversion A vs. B.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Grupo 1:**")
        p1_dp = st.slider("p₁", 0.0, 1.0, 0.6, 0.01, key="p1dp")
        n1_dp = st.slider("n₁", 10, 1000, 100, 1, key="n1dp")
        st.markdown("**Grupo 2:**")
        p2_dp = st.slider("p₂", 0.0, 1.0, 0.5, 0.01, key="p2dp")
        n2_dp = st.slider("n₂", 10, 1000, 100, 1, key="n2dp")
        alpha_dp = st.select_slider("Significancia (α)", [0.01, 0.05, 0.10], value=0.05, key="alphadp")
        tipo_dp = st.radio("Tipo", ["bilateral", "unilateral"], key="tipodp")
    
    with col2:
        beta_dp, pot_dp, lim_inf_dp, lim_sup_dp = potencia_diferencia_proporciones(p1_dp, p2_dp, n1_dp, n2_dp, alpha_dp, tipo_dp)
        dif_dp = p1_dp - p2_dp
        st.metric("Diferencia", f"{dif_dp:.4f}")
        st.metric("Error II (β)", f"{beta_dp:.4f}")
        st.metric("Potencia", f"{pot_dp:.4f}")
        if pot_dp >= 0.80:
            st.success("✅ Adecuada")
        else:
            st.warning("⚠️ Aumentar n")
        h_cohen = 2 * (np.arcsin(np.sqrt(p1_dp)) - np.arcsin(np.sqrt(p2_dp)))
        st.info(f"h = {abs(h_cohen):.3f}")
    
    p_pool_dp = (p1_dp * n1_dp + p2_dp * n2_dp) / (n1_dp + n2_dp)
    se_dp = np.sqrt(p_pool_dp * (1 - p_pool_dp) * (1/n1_dp + 1/n2_dp))
    fig_dp = graficar_distribucion(0, dif_dp, se_dp, lim_inf_dp, lim_sup_dp, tipo_dp, "Diferencia de Proporciones")
    st.pyplot(fig_dp)
    
    df_dp_res = pd.DataFrame([{"p₁": p1_dp, "n₁": n1_dp, "p₂": p2_dp, "n₂": n2_dp, "Dif": round(dif_dp, 4), "α": alpha_dp, "β": round(beta_dp, 4), "Pot": round(pot_dp, 4)}])
    st.dataframe(df_dp_res, use_container_width=True)
    st.download_button("📥 Excel", exportar_excel(df_dp_res), "dif_proporciones.xlsx", key="dldp")

# ==================== TAB 6: VARIANZA ====================
with tabs[5]:
    st.header("Prueba sobre Varianza")
    with st.expander("ℹ️ Informacion"):
        st.markdown("**Cuando usar:** Evaluar si varianza difiere de un valor. Usa Chi-cuadrado.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        s0_var = st.slider("Desv. est. H₀ (σ₀)", 1.0, 50.0, 15.0, 0.1, key="s0var")
        s1_var = st.slider("Desv. est. real (σ₁)", 1.0, 50.0, 20.0, 0.1, key="s1var")
        n_var = st.slider("Muestra (n)", 5, 300, 36, 1, key="nvar")
        alpha_var = st.select_slider("Significancia (α)", [0.01, 0.05, 0.10], value=0.05, key="alphavar")
        tipo_var = st.radio("Tipo", ["bilateral", "unilateral"], key="tipovar")
    
    with col2:
        beta_var, pot_var, lim_inf_var, lim_sup_var = potencia_varianza(s0_var, s1_var, n_var, alpha_var, tipo_var)
        st.metric("Error II (β)", f"{beta_var:.4f}")
        st.metric("Potencia", f"{pot_var:.4f}")
        if pot_var >= 0.80:
            st.success("✅ Adecuada")
        else:
            st.warning("⚠️ Insuficiente")
        st.info(f"σ₁²/σ₀² = {(s1_var/s0_var)**2:.3f}")
    
    df_var_res = pd.DataFrame([{"σ₀": s0_var, "σ₁": s1_var, "n": n_var, "α": alpha_var, "β": round(beta_var, 4), "Potencia": round(pot_var, 4)}])
    st.dataframe(df_var_res, use_container_width=True)
    st.download_button("📥 Excel", exportar_excel(df_var_res), "varianza.xlsx", key="dlvar")

# ==================== TAB 7: RAZON VARIANZAS ====================
with tabs[6]:
    st.header("Razon entre Varianzas (Prueba F)")
    with st.expander("ℹ️ Informacion"):
        st.markdown("**Cuando usar:** Comparar varianzas de 2 poblaciones. Prueba de homogeneidad.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        s1_f = st.slider("Desv. est. grupo 1 (σ₁)", 1.0, 50.0, 15.0, 0.1, key="s1f")
        s2_f = st.slider("Desv. est. grupo 2 (σ₂)", 1.0, 50.0, 10.0, 0.1, key="s2f")
        n1_f = st.slider("n₁", 5, 300, 36, 1, key="n1f")
        n2_f = st.slider("n₂", 5, 300, 36, 1, key="n2f")
        alpha_f = st.select_slider("Significancia (α)", [0.01, 0.05, 0.10], value=0.05, key="alphaf")
    
    with col2:
        beta_f, pot_f, lim_inf_f, lim_sup_f = potencia_razon_varianzas(s1_f, s2_f, n1_f, n2_f, alpha_f)
        st.metric("Error II (β)", f"{beta_f:.4f}")
        st.metric("Potencia", f"{pot_f:.4f}")
        if pot_f >= 0.80:
            st.success("✅ Adecuada")
        else:
            st.warning("⚠️ Insuficiente")
        st.info(f"F = {(s1_f/s2_f)**2:.3f}")
    
    df_f_res = pd.DataFrame([{"σ₁": s1_f, "σ₂": s2_f, "n₁": n1_f, "n₂": n2_f, "α": alpha_f, "β": round(beta_f, 4), "Potencia": round(pot_f, 4)}])
    st.dataframe(df_f_res, use_container_width=True)
    st.download_button("📥 Excel", exportar_excel(df_f_res), "razon_varianzas.xlsx", key="dlf")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>📊 Analisis de Potencia Estadistica - Fundamentos</p>
    <p><small>Herramienta educativa para estadistica basica</small></p>
</div>
""", unsafe_allow_html=True)
