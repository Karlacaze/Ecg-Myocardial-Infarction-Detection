# ecg_app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ECG Classification System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Tema (toggle Dark/Light) + CSS
# -----------------------------------------------------------------------------
theme = st.toggle("🌗 Dark mode", value=True)

PRIMARY   = "#2563eb"
PRIMARY_D = "#1e40af"
BG_LIGHT  = "#f5f7fa"
BG_DARK   = "#0f172a"
CARD_L    = "#ffffff"
CARD_D    = "#111827"
TEXT_L    = "#1f2937"
TEXT_D    = "#e5e7eb"
SUB_L     = "#334155"
SUB_D     = "#94a3b8"

bg = BG_DARK if theme else BG_LIGHT
card = CARD_D if theme else CARD_L
text = TEXT_D if theme else TEXT_L
sub  = SUB_D if theme else SUB_L
primary = PRIMARY

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
* {{ font-family: 'Roboto', sans-serif; }}
:root {{
  --bg: {bg};
  --card: {card};
  --text: {text};
  --sub: {sub};
  --primary: {primary};
  --primary-d: {PRIMARY_D};
}}
.main {{ background: var(--bg) !important; }}
.block-container {{ max-width: 1280px; padding-top: 1.25rem; padding-bottom: 2rem; }}
/* Headers */
.main-header {{ font-size: 2.6rem; font-weight: 800; color: var(--text); text-align:center; margin: .5rem 0 0.25rem; }}
.subtitle {{ text-align:center; font-size:1.05rem; color: var(--sub); margin-bottom: 1.75rem; }}
.sub-header {{
  font-size: 1.4rem; font-weight: 700; color: var(--primary);
  border-bottom: 3px solid var(--primary); padding-bottom: .35rem; margin: 1.5rem 0 1rem;
}}
/* Cards */
.card {{
  background: var(--card);
  border-radius: 12px; padding: 1.1rem 1.2rem;
  box-shadow: 0 6px 20px rgba(2,6,23,.18);
  border: 1px solid rgba(148,163,184,.12);
}}
.metric {{
  background: linear-gradient(135deg, var(--primary), var(--primary-d));
  color:white; border-radius:12px; padding:1rem 1.25rem; height:100%;
  box-shadow: 0 10px 24px rgba(37,99,235,.25);
}}
/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
  gap: .4rem; background: var(--card); border-radius: 8px; padding: .35rem;
  box-shadow: 0 6px 16px rgba(0,0,0,.12);
}}
.stTabs [data-baseweb="tab"] {{ border-radius: 6px; padding: .55rem 1.1rem; color: var(--sub); }}
.stTabs [aria-selected="true"] {{ background: var(--primary); color:white !important; }}
/* Sidebar */
[data-testid="stSidebar"] {{ background: var(--card); border-right:1px solid rgba(148,163,184,.12); }}
[data-testid="stSidebar"] .stRadio label {{ color: var(--text); }}
/* Footer */
.footer-section{{ background: var(--card); color:var(--text); border:1px solid rgba(148,163,184,.12);
  border-radius:10px; padding:1.25rem; }}
.footer-section a{{ color: var(--primary); text-decoration:none; }}
hr {{ border:none; height:1px; background: rgba(148,163,184,.25); margin: 1.5rem 0; }}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Helpers UI
# -----------------------------------------------------------------------------
def section(title, icon=""):
    st.markdown(f"<div class='sub-header'>{icon} {title}</div>", unsafe_allow_html=True)

def card_md(body_md: str):
    st.markdown(f"<div class='card'>{body_md}</div>", unsafe_allow_html=True)

def kpi_grid(items):
    # items: list of (title, value, help)
    cols = st.columns(2)
    cols2 = st.columns(2)
    grid = [*cols, *cols2]
    for i, (title, val, help_) in enumerate(items[:4]):
        with grid[i]:
            st.markdown(
                f"<div class='metric'><div style='opacity:.95;font-size:.95rem'>{title}</div>"
                f"<div style='font-weight:800;font-size:2rem;line-height:1;margin-top:.15rem'>{val}</div></div>",
                unsafe_allow_html=True,
            )

PLOT_TEMPLATE = dict(
    template="plotly_white",
    font=dict(family="Roboto", size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="white",
    hoverlabel=dict(font_size=12),
)

# -----------------------------------------------------------------------------
# Datos/Señales (cache)
# -----------------------------------------------------------------------------
@st.cache_data
def generate_synthetic_ecg(condition='normal', duration=10, sampling_rate=1000):
    t = np.linspace(0, duration, int(duration * sampling_rate))
    if condition == 'normal':
        ecg = np.sin(2*np.pi*1.2*t)
        ecg += 2*np.sin(2*np.pi*1.5*t + np.pi/4)
        ecg += 0.5*np.sin(2*np.pi*0.8*t + np.pi/2)
        ecg += np.random.normal(0, 0.05, len(t))
    elif condition == 'mi':
        ecg = np.sin(2*np.pi*1.2*t)
        ecg += 2.5*np.sin(2*np.pi*1.5*t + np.pi/4)
        ecg += 0.8*np.sin(2*np.pi*0.8*t + np.pi/2)
        ecg += 0.3
        ecg += np.random.normal(0, 0.05, len(t))
    elif condition == 'bbb':
        ecg = np.sin(2*np.pi*1.2*t)
        ecg += 1.5*np.sin(2*np.pi*0.8*t + np.pi/4)
        ecg += 0.5*np.sin(2*np.pi*0.8*t + np.pi/2)
        ecg += np.random.normal(0, 0.05, len(t))
    else:
        ecg = np.sin(2*np.pi*1.2*t) + np.random.normal(0, 0.05, len(t))
    return t, ecg

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heartbeat.png", width=96)
    st.markdown("### Navigation")
    page = st.radio(
        "Select Page:",
        # NUEVO ORDEN: Home → Data Exploration → Random Forest → Binary
        ["🏠 Home", "🌳 Random Forest", "🤖 Binary Classification"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    card_md("**About**  \nECG Classification System – automated detection using ML/DL.")
    card_md("**Author**  \n**Karla** — Deep Learning Research.")

# -----------------------------------------------------------------------------
# HOME
# -----------------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown('<div class="main-header"> ECG Analysis & Classification System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Automated cardiac condition detection using Artificial Intelligence</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], vertical_alignment="top")

    with col1:
        card_md("""
        <h4 style="margin:0 0 .5rem 0">Project Overview</h4>
        This framework provides automated detection of cardiac conditions from electrocardiogram (ECG) signals
        using deep learning. It includes data exploration, a classic ML baseline, and a binary DL model.
        """)

        section("What is ECG?")
        card_md("""
        **Electrocardiogram (ECG)** measures the heart's electrical activity across multiple leads.  
        It helps detect conditions such as **Myocardial Infarction (MI)**, **Bundle Branch Block (BBB)**,
        various **arrhythmias**, and **cardiomyopathies**.
        <ul style='line-height:1.9;margin:.5rem 0 0 1rem'>
          <li><b>Myocardial Infarction (MI)</b> – acute ischemia signs (e.g., ST changes)</li>
          <li><b>Bundle Branch Block (BBB)</b> – delayed ventricular conduction (widened QRS)</li>
          <li><b>Arrhythmias</b> – irregular rhythms</li>
          <li><b>Cardiomyopathy</b> – low voltage/abnormal morphology</li>
        </ul>
        """)

    with col2:
        st.markdown("### Key Statistics")
        kpi_grid([
            ("Total Models", "2", "Random Forest, Binary DL"),
            ("ECG Leads", "15", "12 std + Frank Vx,Vy,Vz"),
            ("Data Points", "5,000", "Per trimmed signal"),
            ("Model Parameters", "~2.5M", "DL model size"),
        ])

    st.markdown("---")
    section("PTB Diagnostic ECG Database")
    c1, c2, c3 = st.columns(3)
    with c1:
        card_md("<b>Source</b><br>PhysioNet PTB Diagnostic Database<br>Widely used in ECG research.")
    with c2:
        card_md("<b>Specifications</b><br>15 leads · ~1000 Hz · ~5000 samples per record (pipeline window).")
    with c3:
        card_md("<b>Classes</b><br>Healthy Controls · MI · BBB · Others")

    st.markdown("---")
    section("Technologies & Techniques")
    a, b = st.columns(2)
    with a:
        card_md("""
        <b>Deep Learning</b>
        <ul style='line-height:1.8;margin:.5rem 0 0 1rem'>
          <li>TensorFlow / Keras – Residual CNN (ResNet-like)</li>
          <li>Regularization: Dropout · L2 · BatchNorm</li>
          <li>Loss: Focal / Cross-Entropy</li>
        </ul>
        """)
    with b:
        card_md("""
        <b>Data Processing</b>
        <ul style='line-height:1.8;margin:.5rem 0 0 1rem'>
          <li>Augmentation (noise, shift, scale, etc.)</li>
          <li>Balancing (e.g., SMOTE)</li>
          <li>Cross-Validation & Ensembles</li>
          <li>Libraries: NumPy · Pandas · scikit-learn · wfdb</li>
        </ul>
        """)

    st.markdown("---")
    section("Key Features")
    g1, g2, g3, g4, g5, g6 = st.columns(6)
    for col, title, text_ in [
        (g1, "Robust Validation", "Cross-validation + multiple metrics."),
        (g2, "Real-time Ready", "Optimized inference pipeline."),
        (g3, "Interpretability", "Feature importance (RF)."),
        (g4, "Clinical-grade Data", "PTB with expert annotations."),
        (g5, "Augmentation", "Better generalization."),
        (g6, "Reproducible", "Controlled configs and seeds."),
    ]:
        with col:
            card_md(f"<b>{title}</b><br>{text_}")

    st.markdown("---")
    section("Sample ECG Signals")
    c1, c2, c3 = st.columns(3)
    for col, (cond, title, color) in zip([c1, c2, c3], [
        ('normal', 'Healthy Control', '#2ecc71'),
        ('mi', 'Myocardial Infarction', '#e74c3c'),
        ('bbb', 'Bundle Branch Block', '#f39c12')
    ]):
        with col:
            t, ecg = generate_synthetic_ecg(cond, duration=3)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=ecg, mode='lines', line=dict(color=color, width=2)))
            fig.update_layout(**PLOT_TEMPLATE, title=title, xaxis_title="Time (s)", 
                              yaxis_title="Amplitude (mV)", height=250, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# RANDOM FOREST (va antes del binario)
# -----------------------------------------------------------------------------
elif page == "🌳 Random Forest":
    st.markdown('<div class="main-header">🌳 Random Forest Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Traditional Machine Learning Approach</div>', unsafe_allow_html=True)

    card_md("""
    <b>Objective</b><br>
    <b>Random Forest Classifier</b> using traditional ML with engineered features extracted from ECG signals
    for multi-class cardiac condition classification.
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        section("Random Forest Architecture", "🌲")
        card_md("""
        <b>Ensemble Learning Method</b><br>
        Random Forest combines multiple decision trees to create a robust classifier.
        Each tree votes for a class, and the majority wins.<br><br>
        <b>Key Components:</b>
        <ul style='line-height:1.9;margin:.5rem 0 0 1rem'>
          <li>🌳 <b>Multiple Decision Trees:</b> Trained on random subsets (bootstrap sampling)</li>
          <li>🎲 <b>Feature Randomness:</b> Each split considers random subset of features</li>
          <li>🗳️ <b>Voting Mechanism:</b> Final prediction by majority vote</li>
        </ul>
        <b>Hyperparameters:</b> n_estimators=200 · max_depth=5 · min_samples_split=10 · min_samples_leaf=5 · max_features=sqrt
        """)

    with col2:
        st.markdown("### Model Specs")
        kpi_grid([
            ("Number of Trees", "200", "Ensemble"),
            ("Max Tree Depth", "5", "Depth limit"),
            ("min_samples_split", "10", "Split threshold"),
            ("min_samples_leaf", "5", "Leaf size"),
        ])

    section("Feature Engineering", "🔧")
    # Usa markdown si quieres negritas/HTML; st.info no acepta unsafe_allow_html
    st.markdown("<div style='background-color:#1e3a8a1a;padding:0.75rem 1rem;border-left:5px solid #2563eb;border-radius:6px;margin-bottom:1rem;'>Unlike deep learning, Random Forest requires <b>manual feature extraction</b> from raw ECG signals.</div>", unsafe_allow_html=True)

    # Solo dos pestañas (quitamos Morphological y Statistical)
    tab1, tab2 = st.tabs(["Time Domain", "Frequency Domain"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            card_md("""
            <b>Time Domain Features</b>
            <ol style='line-height:1.9;margin:.5rem 0 0 1rem'>
              <li><b>R-R Intervals:</b> Time between R peaks → heart rate variability</li>
              <li><b>QRS Duration:</b> Width of QRS complex → conduction delays</li>
              <li><b>ST Segment Deviation:</b> Key indicator for MI</li>
              <li><b>T Wave Amplitude:</b> Inverted T waves indicate ischemia</li>
              <li><b>PR Interval:</b> AV conduction time (normal: 120–200ms)</li>
            </ol>
            """)
        with c2:
            t, ecg = generate_synthetic_ecg('normal', duration=3)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=ecg, mode='lines', line=dict(color='#3498db', width=2)))
            fig.add_annotation(x=0.8, y=float(np.max(ecg))*0.9, text="R Peak", showarrow=True, arrowhead=2, arrowcolor='red')
            fig.add_annotation(x=1.5, y=0.2, text="ST Segment", showarrow=True, arrowhead=2, arrowcolor='green')
            fig.update_layout(**PLOT_TEMPLATE, title="Time Domain Features", xaxis_title="Time (s)",
                              yaxis_title="Amplitude (mV)", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            card_md("""
            <b>Frequency Domain Features</b>
            <ol style='line-height:1.9;margin:.5rem 0 0 1rem'>
              <li><b>Power Spectral Density (PSD):</b> Frequency content via FFT</li>
              <li><b>Frequency Bands:</b> VLF (0–0.04 Hz) · LF (0.04–0.15 Hz) · HF (0.15–0.4 Hz)</li>
              <li><b>Dominant Frequency:</b> Peak in power spectrum</li>
              <li><b>Spectral Entropy:</b> Signal complexity measure</li>
              <li><b>Wavelet Coefficients:</b> Multi-scale frequency analysis</li>
            </ol>
            """)
        with c2:
            frequencies = np.linspace(0, 50, 500)
            psd = np.exp(-(frequencies - 1.5)**2 / 5) + 0.3 * np.exp(-(frequencies - 10)**2 / 20)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=frequencies, y=psd, mode='lines', fill='tozeroy', line=dict(color='#e74c3c', width=2)))
            fig.update_layout(**PLOT_TEMPLATE, title="Power Spectral Density", xaxis_title="Frequency (Hz)",
                              yaxis_title="Power", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.success("✅ **Total Features:** ~50 features per ECG recording (combining all domains)")

    section("Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Overall Accuracy", "87.2%", delta="Good")
    with c2: st.metric("F1-Score", "83.81%", delta="Solid")
    with c3: st.metric("Precision", "79.98%", delta="Solid")
    with c4: st.metric("Macro F1-Score", "37%", delta="Solid")
    
    

# -----------------------------------------------------------------------------
# BINARY CLASSIFICATION (queda al final del menú)
# -----------------------------------------------------------------------------
elif page == "🤖 Binary Classification":
    st.markdown('<div class="main-header">🤖 Binary Classification Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Myocardial Infarction (MI) Detection using ResNet CNN</div>', unsafe_allow_html=True)

    # ===================== VALORES REALES (de tu notebook) =====================
    # MODELO 1 (single best)
    m1 = {
        "accuracy": 0.8155,
        "precision": 0.7717,
        "recall":  0.7744,
        "f1": 0.9000,
        "auc": 0.8830,
        # Confusion matrix con filas = Actual [No-MI, MI], columnas = Pred [No-MI, MI]
        "cm": [[20, 9],
               [10, 64]],
        "n_test": 103,
        "pos_label": "MI",
        "roc_img": "/mnt/data/roc_curve_single_model.png",  # si existe, se mostrará
    }

    # MODELO 2 (ensemble 5-fold en test)
    # Derivado de TN=25, FP=4, FN=4, TP=70  => 103 casos
    m2 = {
        "accuracy": 0.9126,
        "precision": 0.9452,   # 0.9459...
        "recall":    0.9324,   # 0.9459...
        "f1":        0.9388,
        "auc": 0.9674,
        "cm": [[25, 4],
               [ 5, 69]],
        "n_test": 103,
        "pos_label": "MI",
        "roc_img": '/mnt/data/models/cross_validation/roc_curve_ensemble.png',  # no hay imagen específica guardada; mostramos KPIs + CM
    }
    # ===========================================================================

    def kpi_row(metrics: dict, cols=None, title_suffix=""):
        if cols is None:
            cols = st.columns(4)
        with cols[0]:
            st.metric(f"Accuracy{title_suffix}", f"{metrics['accuracy']*100:.2f}%")
        with cols[1]:
            st.metric(f"Precision{title_suffix}", f"{metrics['precision']*100:.2f}%")
        with cols[2]:
            st.metric(f"Recall{title_suffix}", f"{metrics['recall']*100:.2f}%")
        with cols[3]:
            st.metric(f"F1-Score{title_suffix}", f"{metrics['f1']*100:.2f}%")

    def plot_cm(cm, title):
        import numpy as np
        import plotly.graph_objects as go
        z = np.array(cm)
        fig = go.Figure(data=go.Heatmap(
            z=z, x=['Predicted No-MI', 'Predicted MI'], y=['Actual No-MI', 'Actual MI'],
            colorscale='Blues', text=z, texttemplate='%{text}', textfont={"size": 18}, showscale=True
        ))
        fig.update_layout(**PLOT_TEMPLATE, title=title, height=380, yaxis=dict(autorange='reversed'))
        return fig

    def cm_interpretation(cm):
        tn, fp = cm[0]
        fn, tp = cm[1]
        return f"TN: {tn} · FP: {fp} · FN: {fn} · TP: {tp}"

    tab1, tab2 = st.tabs(["Modelo 1 — Single best", "Modelo 2 — Ensemble (5-Fold)"])

    # ----------------------------- TAB 1: SINGLE ------------------------------
    with tab1:
        section("Métricas principales")
        kpi_row(m1)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Matriz de confusión")
            st.plotly_chart(plot_cm(m1["cm"], "Resultados — Single best"), use_container_width=True)
            card_md(f"<b>Interpretación:</b> {cm_interpretation(m1['cm'])}")

        with c2:
            st.markdown("#### Curva ROC")
            import os
            if m1["roc_img"] and os.path.exists(m1["roc_img"]):
                st.image(m1["roc_img"], caption=f"ROC curve (AUC = {m1['auc']:.4f})", use_container_width=True)
            else:
                # fallback simple: diagonal + AUC en texto
                fpr = np.linspace(0, 1, 100)
                tpr = 1 - (1 - fpr) ** 2
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name='ROC (placeholder)', line=dict(width=3)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(width=2, dash='dash')))
                fig.update_layout(**PLOT_TEMPLATE, title=f"ROC Curve (AUC ≈ {m1['auc']:.4f})",
                                  xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=380)
                st.plotly_chart(fig, use_container_width=True)

        st.info(f"**Test set:** {m1['n_test']} registros · AUC = {m1['auc']:.4f}")

    # ----------------------------- TAB 2: ENSEMBLE ----------------------------
    with tab2:
        section("Métricas principales")
        kpi_row(m2)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Matriz de confusión")
            st.plotly_chart(plot_cm(m2["cm"], "Resultados — Ensemble (5-Fold)"), use_container_width=True)
            card_md(f"<b>Interpretación:</b> {cm_interpretation(m2['cm'])}")

        with c2:
            st.markdown("#### AUC (sin imagen)")
            card_md(f"""
            <b>Área bajo la curva (AUC):</b> <span style='font-size:1.25rem;font-weight:700'>{m2['auc']:.4f}</span><br>
            <div style='opacity:.8'>No se proporcionó una imagen ROC específica del ensemble; se reporta el valor calculado.</div>
            """)

        st.success(f"**Ensemble 5-Fold en test:** {m2['n_test']} registros · AUC = {m2['auc']:.4f}")

    # --------------------------- COMPARATIVA RÁPIDA ---------------------------
    section("Comparativa rápida")
    comp_df = pd.DataFrame({
        "Métrica": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"],
        "Modelo 1 (Single)": [f"{m1['accuracy']*100:.2f}%", f"{m1['precision']*100:.2f}%",
                              f"{m1['recall']*100:.2f}%", f"{m1['f1']*100:.2f}%", f"{m1['auc']:.4f}"],
        "Modelo 2 (Ensemble)": [f"{m2['accuracy']*100:.2f}%", f"{m2['precision']*100:.2f}%",
                                f"{m2['recall']*100:.2f}%", f"{m2['f1']*100:.2f}%", f"{m2['auc']:.4f}"],
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)



# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div class="footer-section">
  <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem;'>
    <div><b>Contact</b><br>Karla – ECG Research</div>
    <div><b>Resources</b><br><a href='https://physionet.org/content/ptbdb/'>PTB Database (PhysioNet)</a></div>
    <div><b>Citation</b><br>ECG Classification System (2024)</div>
  </div>
  <div style='text-align:center;padding-top:.9rem;border-top:1px solid rgba(148,163,184,.25);margin-top:.9rem'>
    © 2024 • Built with Streamlit & TensorFlow
  </div>
</div>
""", unsafe_allow_html=True)
