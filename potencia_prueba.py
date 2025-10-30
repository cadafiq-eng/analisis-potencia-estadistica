import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, f
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

# Funciones de cálculo
def potencia_media(mu0, mu1, sigma, n, alpha, tipo='bilateral'):
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

def potencia_proporcion(p0, p1, n, alpha, tipo='bilateral'):
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
    ax.set_xlabel('Valor del estadístico', fontsize=12)
    ax.set_ylabel('Densidad', fontsize=12)
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.close()
    return fig

def graficar_curva_potencia(mu0, sigma, n, alpha, tipo='bilateral'):
    fig, ax = plt.subplots(figsize=(10, 6))
    mu1_values = np.linspace(mu0 - 3*sigma/np.sqrt(n), mu0 + 3*sigma/np.sqrt(n), 100)
    potencias = []
    for mu1 in mu1_values:
        if abs(mu1 - mu0) < 0.001:
            potencias.append(alpha)
        else:
            beta, potencia, _, _ = potencia_media(mu0, mu1, sigma, n, alpha, tipo)
            potencias.append(potencia)
    ax.plot(mu1_values, potencias, linewidth=2, color='darkblue')
    ax.axhline(y=alpha, color='red', linestyle='--', label=f'α = {alpha}')
    ax.axhline(y=0.80, color='green', linestyle='--', label='Potencia 0.80')
    ax.axvline(x=mu0, color='gray', linestyle=':', label=f'μ₀ = {mu0}')
    ax.set_xlabel('Media verdadera', fontsize=12)
    ax.set_ylabel('Potencia', fontsize=12)
    ax.set_title('Curva de Potencia', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
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

# Interfaz
st.set_page_config(page_title="Analisis de Potencia", layout="wide", page_icon="📊")

with st.sidebar:
    st.title("📚 Guia Rapida")
    st.markdown("""
    ### Conceptos Clave
    **Error Tipo I (α):** Rechazar H₀ cuando es verdadera
    **Error Tipo II (β):** No rechazar H₀ cuando es falsa
    **Potencia (1-β):** Rechazar H₀ cuando es falsa (ideal ≥ 0.80)
    """)

st.title("🔍 Analisis de Potencia y Error Tipo II")
st.markdown("Explora como diferentes factores afectan la potencia estadistica.")

tabs = st.tabs(["📐 Media", "📊 Proporciones", "📈 Varianza", "🔄 Razon de Varianzas", "⚖️ Comparacion"])

with tabs[0]:
    st.header("Prueba sobre la Media")
    with st.expander("ℹ️ Informacion"):
        st.markdown("Prueba si la media poblacional difiere de un valor especifico.")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Parametros")
        mu0 = st.slider("Media bajo H₀ (μ₀)", 50.0, 150.0, 100.0, 0.1)
        mu1 = st.slider("Media verdadera (μ₁)", 50.0, 150.0, 105.0, 0.1)
        sigma = st.slider("Desviacion estandar (σ)", 1.0, 50.0, 15.0, 0.1)
        n = st.slider("Tamano de muestra (n)", 5, 500, 36, 1)
        alpha = st.select_slider("Nivel de significancia (α)", options=[0.01, 0.05, 0.10], value=0.05)
        tipo = st.radio("Tipo de prueba", ["bilateral", "unilateral"])
    with col2:
        st.subheader("Resultados")
        beta, potencia, lim_inf, lim_sup = potencia_media(mu0, mu1, sigma, n, alpha, tipo)
        st.metric("Error Tipo II (β)", f"{beta:.4f}")
        st.metric("Potencia (1-β)", f"{potencia:.4f}")
        if potencia >= 0.80:
            st.success("✅ Potencia adecuada")
        elif potencia >= 0.60:
            st.warning("⚠️ Potencia moderada")
        else:
            st.error("❌ Potencia baja")
        st.info(f"Tamano del efecto: d = {abs(mu1-mu0)/sigma:.3f}")
    st.subheader("Visualizacion")
    fig1 = graficar_distribucion(mu0, mu1, sigma/np.sqrt(n), lim_inf, lim_sup, tipo, "Distribuciones")
    st.pyplot(fig1)
    st.subheader("Curva de potencia")
    fig2 = graficar_curva_potencia(mu0, sigma, n, alpha, tipo)
    st.pyplot(fig2)
    df_result = pd.DataFrame([{"μ₀": mu0, "μ₁": mu1, "σ": sigma, "n": n, "α": alpha, "β": round(beta, 4), "Potencia": round(potencia, 4)}])
    st.dataframe(df_result, use_container_width=True)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button("📥 Excel", data=exportar_excel(df_result), file_name="potencia_media.xlsx")
    with col_d2:
        texto_pdf = f"Potencia: {potencia:.4f}\nBeta: {beta:.4f}\nmu0: {mu0}\nmu1: {mu1}\nn: {n}\nalpha: {alpha}"
        st.download_button("📄 PDF", data=exportar_pdf(texto_pdf), file_name="analisis_media.pdf")

with tabs[1]:
    st.header("Prueba sobre Proporciones")
    with st.expander("ℹ️ Informacion"):
        st.markdown("Prueba si una proporcion poblacional difiere de un valor especifico.")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Parametros")
        p0 = st.slider("Proporcion bajo H₀ (p₀)", 0.0, 1.0, 0.5, 0.01)
        p1 = st.slider("Proporcion verdadera (p₁)", 0.0, 1.0, 0.6, 0.01)
        n_prop = st.slider("Tamano de muestra (n)", 10, 1000, 100, 1)
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
    fig_prop = graficar_distribucion(p0, p1, se_prop, lim_inf_prop, lim_sup_prop, tipo_prop, "Distribuciones")
    st.pyplot(fig_prop)
    df_prop = pd.DataFrame([{"p₀": p0, "p₁": p1, "n": n_prop, "α": alpha_prop, "β": round(beta_prop, 4), "Potencia": round(potencia_prop, 4)}])
    st.dataframe(df_prop, use_container_width=True)
    st.download_button("📥 Excel", data=exportar_excel(df_prop), file_name="potencia_proporcion.xlsx", key="dl_prop")

with tabs[2]:
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
    df_var = pd.DataFrame([{"σ₀": sigma0_var, "σ₁": sigma1_var, "n": n_var, "α": alpha_var, "β": round(beta_var, 4), "Potencia": round(potencia_var, 4)}])
    st.dataframe(df_var, use_container_width=True)
    st.download_button("📥 Excel", data=exportar_excel(df_var), file_name="potencia_varianza.xlsx", key="dl_var")

with tabs[3]:
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
    df_f = pd.DataFrame([{"σ₁": sigma1_f, "σ₂": sigma2_f, "n₁": n1_f, "n₂": n2_f, "α": alpha_f, "β": round(beta_f, 4), "Potencia": round(potencia_f, 4)}])
    st.dataframe(df_f, use_container_width=True)
    st.download_button("📥 Excel", data=exportar_excel(df_f), file_name="potencia_razon_varianzas.xlsx", key="dl_f")

with tabs[4]:
    st.header("Comparacion de Escenarios")
    st.markdown("Compara diferentes configuraciones para entender como cambian la potencia y beta.")
    tipo_comparacion = st.selectbox("Tipo de prueba:", ["Media", "Proporciones", "Varianza"])
    num_escenarios = st.slider("Numero de escenarios", 2, 5, 3)
    escenarios = []
    if tipo_comparacion == "Media":
        cols = st.columns(num_escenarios)
        for i in range(num_escenarios):
            with cols[i]:
                st.subheader(f"Escenario {i+1}")
                mu0_c = st.number_input(f"μ₀", value=100.0, key=f"mu0c{i}")
                mu1_c = st.number_input(f"μ₁", value=105.0+i*2, key=f"mu1c{i}")
                sigma_c = st.number_input(f"σ", value=15.0, key=f"sc{i}")
                n_c = st.number_input(f"n", value=36+i*10, key=f"nc{i}", min_value=5)
                alpha_c = st.selectbox(f"α", [0.01, 0.05, 0.10], index=1, key=f"ac{i}")
                beta, potencia, _, _ = potencia_media(mu0_c, mu1_c, sigma_c, int(n_c), alpha_c, 'bilateral')
                escenarios.append({"Escenario": i+1, "μ₀": mu0_c, "μ₁": mu1_c, "σ": sigma_c, "n": int(n_c), "α": alpha_c, "β": round(beta, 4), "Potencia": round(potencia, 4)})
    elif tipo_comparacion == "Proporciones":
        cols = st.columns(num_escenarios)
        for i in range(num_escenarios):
            with cols[i]:
                st.subheader(f"Escenario {i+1}")
                p0_c = st.number_input(f"p₀", value=0.5, min_value=0.0, max_value=1.0, key=f"p0c{i}")
                p1_c = st.number_input(f"p₁", value=0.5+(i+1)*0.05, min_value=0.0, max_value=1.0, key=f"p1c{i}")
                n_c = st.number_input(f"n", value=100+i*50, key=f"npc{i}", min_value=10)
                alpha_c = st.selectbox(f"α", [0.01, 0.05, 0.10], index=1, key=f"apc{i}")
                beta, potencia, _, _ = potencia_proporcion(p0_c, p1_c, int(n_c), alpha_c, 'bilateral')
                escenarios.append({"Escenario": i+1, "p₀": p0_c, "p₁": p1_c, "n": int(n_c), "α": alpha_c, "β": round(beta, 4), "Potencia": round(potencia, 4)})
    else:
        cols = st.columns(num_escenarios)
        for i in range(num_escenarios):
            with cols[i]:
                st.subheader(f"Escenario {i+1}")
                sigma0_c = st.number_input(f"σ₀", value=15.0, key=f"s0c{i}")
                sigma1_c = st.number_input(f"σ₁", value=20.0+i*2, key=f"s1c{i}")
                n_c = st.number_input(f"n", value=36+i*10, key=f"nvc{i}", min_value=5)
                alpha_c = st.selectbox(f"α", [0.01, 0.05, 0.10], index=1, key=f"avc{i}")
                beta, potencia, _, _ = potencia_varianza(sigma0_c, sigma1_c, int(n_c), alpha_c, 'bilateral')
                escenarios.append({"Escenario": i+1, "σ₀": sigma0_c, "σ₁": sigma1_c, "n": int(n_c), "α": alpha_c, "β": round(beta, 4), "Potencia": round(potencia, 4)})
    st.subheader("Tabla Comparativa")
    df_comp = pd.DataFrame(escenarios)
    st.dataframe(df_comp, use_container_width=True)
    st.subheader("Comparacion Visual")
    fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    escenarios_num = [e["Escenario"] for e in escenarios]
    potencias = [e["Potencia"] for e in escenarios]
    betas = [e["β"] for e in escenarios]
    ax1.bar(escenarios_num, potencias, color='green', alpha=0.7)
    ax1.axhline(y=0.80, color='red', linestyle='--', label='Objetivo 0.80')
    ax1.set_xlabel('Escenario')
    ax1.set_ylabel('Potencia')
    ax1.set_title('Comparacion de Potencia')
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    x = np.arange(len(escenarios_num))
    width = 0.35
    ax2.bar(x - width/2, [e["α"] for e in escenarios], width, label='Error Tipo I (α)', color='blue', alpha=0.7)
    ax2.bar(x + width/2, betas, width, label='Error Tipo II (β)', color='orange', alpha=0.7)
    ax2.set_xlabel('Escenario')
    ax2.set_ylabel('Probabilidad')
    ax2.set_title('Comparacion de Errores')
    ax2.set_xticks(x)
    ax2.set_xticklabels(escenarios_num)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.close()
    st.pyplot(fig_comp)
    mejor = max(escenarios, key=lambda x: x["Potencia"])
    peor = min(escenarios, key=lambda x: x["Potencia"])
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.success(f"✅ Mejor: Escenario {mejor['Escenario']} (Potencia: {mejor['Potencia']:.4f})")
    with col_a2:
        st.warning(f"⚠️ Menor: Escenario {peor['Escenario']} (Potencia: {peor['Potencia']:.4f})")
    st.markdown("### Recomendaciones")
    recomendaciones = []
    for esc in escenarios:
        if esc["Potencia"] < 0.80:
            if tipo_comparacion == "Media":
                n_req = int(((norm.ppf(1-esc["α"]/2) + norm.ppf(0.80)) * esc["σ"] / abs(esc["μ₁"] - esc["μ₀"]))**2) + 1
                recomendaciones.append(f"- Escenario {esc['Escenario']}: Aumentar n a ~{n_req}")
            else:
                recomendaciones.append(f"- Escenario {esc['Escenario']}: Aumentar tamano de muestra")
    if recomendaciones:
        for rec in recomendaciones:
            st.markdown(rec)
    else:
        st.success("✅ Todos los escenarios tienen potencia adecuada")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.download_button("📥 Excel", data=exportar_excel(df_comp), file_name="comparacion.xlsx", key="dl_comp")
    with col_e2:
        texto_pdf_comp = "\n".join([f"Escenario {e['Escenario']}: Potencia={e['Potencia']}, Beta={e['β']}" for e in escenarios])
        st.download_button("📄 PDF", data=exportar_pdf(texto_pdf_comp), file_name="comparacion.pdf", key="dl_comp_pdf")

st.markdown("---")
st.markdown('<div style="text-align: center; color: gray;"><p>📊 Analisis de Potencia Estadistica | Streamlit</p><p><small>Herramienta educativa</small></p></div>', unsafe_allow_html=True)
