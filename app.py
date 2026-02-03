import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIAHA — Flux Touristiques Maroc",
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Police */
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Fond principal */
  .stApp { background-color: #F7F4EF; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #1A1209 !important;
    border-right: 2px solid #C4562A;
  }
  [data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stRadio label { color: rgba(255,255,255,0.6) !important; font-size:.78rem !important; letter-spacing:.05em; text-transform:uppercase; }

  /* Métriques */
  [data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid rgba(26,18,9,.09);
    border-radius: 12px;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 2px 16px rgba(26,18,9,.05);
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: #1A1209 !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: .75rem !important;
    font-weight: 600 !important;
    letter-spacing: .05em;
    color: #888 !important;
    text-transform: uppercase;
  }

  /* Titres */
  h1 { font-family:'Syne',sans-serif !important; font-weight:800 !important; color:#1A1209 !important; }
  h2 { font-family:'Syne',sans-serif !important; font-weight:700 !important; color:#1A1209 !important; }
  h3 { font-family:'Syne',sans-serif !important; font-weight:700 !important; color:#1A1209 !important; }

  /* Cards personnalisées */
  .siaha-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    border: 1px solid rgba(26,18,9,.08);
    box-shadow: 0 2px 20px rgba(26,18,9,.05);
    margin-bottom: 1rem;
  }
  .result-high {
    background: linear-gradient(135deg, #FFF0EB, #FFE4D9);
    border: 2px solid #C4562A;
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
  }
  .result-low {
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    border: 2px solid #2A5C8A;
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
  }
  .badge {
    display:inline-block; padding:.25rem .75rem; border-radius:20px;
    font-size:.72rem; font-weight:700; letter-spacing:.05em; text-transform:uppercase;
  }
  .badge-terr { background:#C4562A22; color:#C4562A; }
  .badge-blue { background:#2A5C8A22; color:#2A5C8A; }
  .badge-gold { background:#D4A83222; color:#A67F10; }

  /* Divider */
  hr { border-color: rgba(26,18,9,.1) !important; }

  /* Bouton principal */
  .stButton > button {
    background: #C4562A !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: 'Syne', sans-serif !important;
    letter-spacing: .04em !important;
    padding: .65rem 2rem !important;
    width: 100% !important;
    font-size: 1rem !important;
    transition: all .2s !important;
  }
  .stButton > button:hover {
    background: #E8784E !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px #C4562A44 !important;
  }

  /* Selectbox */
  .stSelectbox > div > div {
    background: #FAF8F4 !important;
    border: 1.5px solid rgba(26,18,9,.2) !important;
    border-radius: 8px !important;
  }

  /* Tab */
  .stTabs [data-baseweb="tab-list"] {
    background: #F0EBE3 !important;
    border-radius: 10px;
    padding: 3px;
    gap: 2px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: .85rem !important;
  }
  .stTabs [aria-selected="true"] {
    background: #1A1209 !important;
    color: white !important;
  }
</style>
""", unsafe_allow_html=True)


# ─── CHARGEMENT DONNÉES ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_nat  = pd.read_excel("evolution-par-nationalite-des-arrivees-des-touristes-aux-postes-frontieres.xlsx")
    df_prov = pd.read_excel("evolution-par-point-dentree-des-arrivees-des-touristes-aux-postes-frontieres.xlsx")
    df_nuit = pd.read_excel("evolution-nuitees-realisees-etablissements-hebergement-touristique-par-destination.xlsx")
    return df_nat, df_prov, df_nuit

@st.cache_resource
def load_model():
    model    = joblib.load("model_naivebayes.pkl")
    le_nat   = joblib.load("le_nat.pkl")
    le_prov  = joblib.load("le_prov.pkl")
    le_trim  = joblib.load("le_trim.pkl")
    le_inter = joblib.load("le_inter.pkl")
    return model, le_nat, le_prov, le_trim, le_inter

df_nat, df_prov, df_nuit = load_data()
model, le_nat, le_prov, le_trim, le_inter = load_model()

YEARS = [str(y) for y in range(2012, 2021)]
COLORS = ["#C4562A","#2A5C8A","#D4A832","#5B9E6F","#8B4FB5",
          "#E8784E","#4A82B8","#E8C84E","#7DCE96","#B47FE0",
          "#A63E1C","#1A3F62","#A6801A"]

POSTE_LABELS = {
    "A Agadir Almassira":      "✈ Agadir Al Massira",
    "A Al Hoceima":            "✈ Al Hoceima",
    "A Essaouira":             "✈ Essaouira",
    "A Fes-Saiss":             "✈ Fès-Saïss",
    "A Laaroui":               "✈ Laaroui (Nador)",
    "A Marrakech Ménara":      "✈ Marrakech Ménara",
    "A Mohammed V":            "✈ Mohammed V (Casablanca)",
    "A Oaurzazate":            "✈ Ouarzazate",
    "A Oujda":                 "✈ Oujda",
    "A Rabat-Salé":            "✈ Rabat-Salé",
    "A Tanger Ibn Battouta":   "✈ Tanger Ibn Battouta",
    "P Nador":                 "🚢 Port de Nador",
    "P Tanger":                "🚢 Port de Tanger",
    "P Tanger Med":            "🚢 Tanger Med",
    "T Bab Sebta":             "🛣 Bab Sebta",
    "T Béni Anzar":            "🛣 Béni Anzar",
    "T.AIR":                   "✈ Total Aérien",
    "T.MER":                   "🚢 Total Maritime",
    "T.TERRE":                 "🛣 Total Terrestre",
}
LABEL_TO_CODE = {v: k for k, v in POSTE_LABELS.items()}

NAT_FLAGS = {
    "Touristes Etrangers": "🌍", "France": "🇫🇷", "Espagne": "🇪🇸",
    "Royaume-Uni": "🇬🇧", "Allemagne": "🇩🇪", "Italie": "🇮🇹",
    "Etats Unis": "🇺🇸", "Belgique": "🇧🇪", "Hollande": "🇳🇱",
    "Maghreb": "🌐", "Chine": "🇨🇳", "Scandinavie": "🌊", "MRE": "🇲🇦"
}
TRIM_LABELS = {1: "T1 — Janvier à Mars", 2: "T2 — Avril à Juin",
               3: "T3 — Juillet à Septembre", 4: "T4 — Octobre à Décembre"}


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 .5rem'>
      <div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;color:white'>
        S<span style='color:#E8784E'>IA</span>HA
      </div>
      <div style='font-size:.7rem;color:rgba(255,255,255,.4);letter-spacing:.1em;margin-top:.2rem'>
        SYSTÈME D'ANALYSE DES FLUX TOURISTIQUES
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='color:rgba(255,255,255,.4);font-size:.7rem;letter-spacing:.1em'>NAVIGATION</p>", unsafe_allow_html=True)

    page = st.radio(
        "",
        ["🏠  Tableau de bord", "🔮  Simulateur IA", "📊  Analyse des données", "⚙  Méthodologie"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:.72rem;color:rgba(255,255,255,.35);line-height:1.8'>
      <div>📅 Données : 2012–2020</div>
      <div>🤖 Modèle : Naive Bayes Multinomial</div>
      <div>🏆 Précision : ~85%</div>
      <div>🌍 Objectif : Mondial 2030</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TABLEAU DE BORD
# ══════════════════════════════════════════════════════════════════════════════
if "Tableau" in page:
    st.markdown("""
    <h1 style='margin-bottom:.25rem'>Tableau de bord</h1>
    <p style='color:#888;margin-bottom:2rem'>Aperçu général du tourisme marocain · ONMT 2012–2020</p>
    """, unsafe_allow_html=True)

    # KPIs
    total_2019 = int(df_nat[df_nat['nationalite']=='Touristes Etrangers'][2019].values[0])
    total_2012 = int(df_nat[df_nat['nationalite']=='Touristes Etrangers'][2012].values[0])
    total_2020 = int(df_nat[df_nat['nationalite']=='Touristes Etrangers'][2020].values[0])
    top_nat = df_nat[df_nat['nationalite']!='Touristes Etrangers'].nlargest(1, 2019)['nationalite'].values[0]
    top_prov = df_prov.nlargest(1, 2019)['provinces'].values[0]
    top_nuit = df_nuit.nlargest(1, 2019)['provinces'].values[0]
    covid_drop = round((1 - total_2020/total_2019)*100, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Touristes 2019", f"{total_2019/1e6:.2f}M", f"+{(total_2019/total_2012-1)*100:.0f}% vs 2012")
    c2.metric("Impact COVID 2020", f"{total_2020/1e6:.2f}M", f"-{covid_drop}% vs 2019", delta_color="inverse")
    c3.metric("1ère nationalité 2019", NAT_FLAGS.get(top_nat,'') + " " + top_nat)
    c4.metric("1er aéroport 2019", top_prov.replace("A ","✈ "))

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("#### Évolution des arrivées (2012–2020)")
        df_plot = df_nat[df_nat['nationalite'] != 'Touristes Etrangers'].copy()
        df_melt = df_plot.melt(id_vars='nationalite', value_vars=list(range(2012,2021)), var_name='Année', value_name='Arrivées')
        df_melt['Année'] = df_melt['Année'].astype(str)
        fig = px.line(df_melt, x='Année', y='Arrivées', color='nationalite',
                      color_discrete_sequence=COLORS,
                      labels={'nationalite':'Nationalité','Arrivées':'Arrivées'},
                      template='simple_white')
        fig.update_traces(line_width=2.2)
        fig.update_layout(
            height=360, legend=dict(orientation='h', y=-0.3, font_size=11),
            margin=dict(l=0,r=0,t=10,b=0),
            yaxis_tickformat='.2s', paper_bgcolor='white', plot_bgcolor='white',
            font_family='Inter'
        )
        fig.update_xaxes(showgrid=True, gridcolor='#F0EBE3')
        fig.update_yaxes(showgrid=True, gridcolor='#F0EBE3')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Parts de marché 2019")
        df_pie = df_nat[df_nat['nationalite'] != 'Touristes Etrangers'][['nationalite', 2019]].copy()
        fig2 = px.pie(df_pie, names='nationalite', values=2019,
                      color_discrete_sequence=COLORS, hole=0.45,
                      template='simple_white')
        fig2.update_traces(textposition='inside', textinfo='percent', textfont_size=10)
        fig2.update_layout(
            height=360, showlegend=True,
            legend=dict(font_size=10, orientation='v'),
            margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor='white', font_family='Inter'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Nuitées bar
    st.markdown("#### Nuitées par destination (2012–2020)")
    df_nuit_melt = df_nuit.melt(id_vars='provinces', value_vars=list(range(2012,2021)), var_name='Année', value_name='Nuitées')
    df_nuit_melt['Année'] = df_nuit_melt['Année'].astype(str)
    fig3 = px.bar(df_nuit_melt, x='Année', y='Nuitées', color='provinces',
                  color_discrete_sequence=COLORS, barmode='group',
                  template='simple_white',
                  labels={'provinces':'Destination'})
    fig3.update_layout(
        height=340, legend=dict(orientation='h', y=-0.3, font_size=11),
        margin=dict(l=0,r=0,t=10,b=0),
        yaxis_tickformat='.2s', paper_bgcolor='white', plot_bgcolor='white',
        font_family='Inter'
    )
    fig3.update_xaxes(showgrid=False)
    fig3.update_yaxes(gridcolor='#F0EBE3')
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SIMULATEUR IA
# ══════════════════════════════════════════════════════════════════════════════
elif "Simulateur" in page:
    st.markdown("""
    <h1 style='margin-bottom:.25rem'>🔮 Simulateur IA</h1>
    <p style='color:#888;margin-bottom:2rem'>Prédiction Naive Bayes du flux touristique aux postes-frontières</p>
    """, unsafe_allow_html=True)

    col_form, col_res = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="siaha-card">', unsafe_allow_html=True)
        st.markdown("**⚙ Paramètres de simulation**")
        st.markdown("<br>", unsafe_allow_html=True)

        nat_options = [f"{NAT_FLAGS.get(n,'')} {n}" for n in le_nat.classes_]
        nat_label = st.selectbox("🌍 Nationalité du touriste", nat_options, index=5)
        nat_code  = nat_label.split(" ", 1)[1].strip()

        prov_options = [POSTE_LABELS.get(p, p) for p in le_prov.classes_]
        prov_label = st.selectbox("📍 Poste-frontière / Destination", prov_options, index=10)
        prov_code  = LABEL_TO_CODE.get(prov_label, prov_label)

        trim = st.select_slider(
            "📅 Trimestre",
            options=[1, 2, 3, 4],
            value=3,
            format_func=lambda x: TRIM_LABELS[x]
        )

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Calculer la prédiction →")
        st.markdown('</div>', unsafe_allow_html=True)

        # Info sur les features
        with st.expander("ℹ️ Comprendre les variables d'entrée"):
            st.markdown("""
            Le modèle utilise **4 features encodées** :
            - `nat_encoded` — Nationalité (LabelEncoder)
            - `prov_encoded` — Poste-frontière (LabelEncoder)
            - `trim_encoded` — Trimestre 1–4 (saisonnalité)
            - `inter_encoded` — Interaction Nat × Province (feature engineering)

            **Cible** : `target_flux` = 1 si flux > médiane historique, 0 sinon.
            """)

    with col_res:
        if run:
            try:
                nat_idx  = le_nat.transform([nat_code])[0]
                prov_idx = le_prov.transform([prov_code])[0]
                trim_idx = le_trim.transform([trim])[0]
                inter_str = f"{nat_idx}_{prov_idx}"
                inter_idx = le_inter.transform([inter_str])[0]

                proba = model.predict_proba(
                    pd.DataFrame([[nat_idx, prov_idx, trim_idx, inter_idx]],
                                 columns=['nat_encoded','prov_encoded','trim_encoded','inter_encoded'])
                )[0]
                p_high = proba[1] * 100
                p_low  = proba[0] * 100
                is_high = p_high >= 50

                css_class = "result-high" if is_high else "result-low"
                icon  = "⚠️" if is_high else "✅"
                badge = f'<span class="badge badge-terr">FLUX ÉLEVÉ</span>' if is_high else f'<span class="badge badge-blue">FLUX MODÉRÉ</span>'

                st.markdown(f"""
                <div class="{css_class}">
                  <div style='font-size:.75rem;font-weight:700;letter-spacing:.08em;color:#888;text-transform:uppercase;margin-bottom:.5rem'>
                    Résultat — {NAT_FLAGS.get(nat_code,'')} {nat_code} → {POSTE_LABELS.get(prov_code, prov_code)} · {TRIM_LABELS[trim]}
                  </div>
                  <div style='font-family:Syne,sans-serif;font-size:2.4rem;font-weight:800;margin:.5rem 0'>{icon} {p_high:.1f}%</div>
                  <div>Probabilité d'un <strong>flux élevé</strong> &nbsp;{badge}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=p_high,
                    number={'suffix': '%', 'font': {'size': 36, 'family': 'Syne', 'color': '#1A1209'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#888', 'tickfont': {'size': 10}},
                        'bar': {'color': "#C4562A" if is_high else "#2A5C8A", 'thickness': 0.25},
                        'bgcolor': "#F7F4EF",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 40],  'color': '#EFF6FF'},
                            {'range': [40, 70], 'color': '#FFF7E6'},
                            {'range': [70, 100],'color': '#FFF0EB'}
                        ],
                        'threshold': {'line': {'color': "#C4562A", 'width': 3}, 'thickness': 0.8, 'value': 70}
                    }
                ))
                fig_gauge.update_layout(
                    height=240, margin=dict(l=20,r=20,t=20,b=10),
                    paper_bgcolor='white', font_family='Inter'
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Decision
                if p_high > 70:
                    msg = "**⚠️ Action recommandée — Renforcer l'infrastructure**\n\nProbabilité de congestion élevée. Mobiliser des ressources supplémentaires, renforcer la sécurité et fluidifier les contrôles à ce poste pour le Mondial 2030."
                elif p_high > 50:
                    msg = "**🔶 Vigilance conseillée — Flux intermédiaire**\n\nLe flux prévu est au-dessus de la médiane. Prévoir des renforts ponctuels et surveiller l'évolution en temps réel."
                else:
                    msg = "**✅ Situation normale — Flux maîtrisable**\n\nLe flux prévu reste dans une fourchette gérable avec les ressources habituelles. Maintenir la surveillance standard."

                st.info(msg)

                # Probabilités détaillées
                c1, c2 = st.columns(2)
                c1.metric("P(Flux Élevé)", f"{p_high:.1f}%", delta=None)
                c2.metric("P(Flux Faible)", f"{p_low:.1f}%", delta=None)

            except ValueError as e:
                st.error(f"❌ Combinaison nationalité × province non présente dans les données d'entraînement. Essayez une autre combinaison.\n\nDétail : `{e}`")
        else:
            st.markdown("""
            <div style='background:white;border-radius:14px;padding:3rem 2rem;text-align:center;border:1px dashed rgba(26,18,9,.15);min-height:400px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1rem'>
              <div style='font-size:3rem'>🔮</div>
              <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#1A1209'>Prêt à simuler</div>
              <div style='color:#888;font-size:.9rem'>Renseignez les paramètres et<br>cliquez sur <strong>Calculer</strong>.</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Simulation en masse ──
    st.markdown("---")
    st.markdown("#### Simulation multi-nationalités pour un poste donné")

    col_mp, col_mt = st.columns([1, 2])
    with col_mp:
        multi_prov_label = st.selectbox("Poste-frontière", prov_options, index=10, key="multi_prov")
        multi_prov_code  = LABEL_TO_CODE.get(multi_prov_label, multi_prov_label)
        multi_trim = st.select_slider("Trimestre", options=[1,2,3,4], value=3,
                                      format_func=lambda x: f"T{x}", key="multi_trim")

    with col_mt:
        rows = []
        for nat_c in le_nat.classes_:
            try:
                ni = le_nat.transform([nat_c])[0]
                pi = le_prov.transform([multi_prov_code])[0]
                ti = le_trim.transform([multi_trim])[0]
                ii = le_inter.transform([f"{ni}_{pi}"])[0]
                proba_m = model.predict_proba(
                    pd.DataFrame([[ni, pi, ti, ii]],
                                 columns=['nat_encoded','prov_encoded','trim_encoded','inter_encoded'])
                )[0]
                rows.append({'Nationalité': f"{NAT_FLAGS.get(nat_c,'')} {nat_c}", 'P(Flux Élevé)': round(proba_m[1]*100,1)})
            except:
                pass

        df_multi = pd.DataFrame(rows).sort_values('P(Flux Élevé)', ascending=True)
        colors_bar = ['#C4562A' if v >= 70 else '#D4A832' if v >= 50 else '#2A5C8A'
                      for v in df_multi['P(Flux Élevé)']]
        fig_bar = go.Figure(go.Bar(
            x=df_multi['P(Flux Élevé)'], y=df_multi['Nationalité'],
            orientation='h', marker_color=colors_bar,
            text=df_multi['P(Flux Élevé)'].apply(lambda x: f"{x:.0f}%"),
            textposition='outside'
        ))
        fig_bar.add_vline(x=70, line_dash="dash", line_color="#C4562A", opacity=.5,
                          annotation_text="Seuil critique 70%", annotation_position="top right")
        fig_bar.update_layout(
            height=340, margin=dict(l=0,r=40,t=10,b=0),
            xaxis=dict(range=[0,110], showgrid=True, gridcolor='#F0EBE3'),
            yaxis=dict(showgrid=False), paper_bgcolor='white',
            plot_bgcolor='white', font_family='Inter', font_size=11
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYSE DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════
elif "Analyse" in page:
    st.markdown("""
    <h1 style='margin-bottom:.25rem'>📊 Analyse des données</h1>
    <p style='color:#888;margin-bottom:2rem'>Exploration interactive des données ONMT 2012–2020</p>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌍 Par nationalité", "🗺 Par poste-frontière", "🏨 Nuitées"])

    # ── TAB 1 : NATIONALITÉS ──
    with tab1:
        c_sel, c_yr = st.columns([2, 1])
        with c_sel:
            nats_sel = st.multiselect(
                "Nationalités à afficher",
                options=df_nat[df_nat['nationalite']!='Touristes Etrangers']['nationalite'].tolist(),
                default=['France', 'Espagne', 'MRE', 'Royaume-Uni', 'Maghreb']
            )
        with c_yr:
            yr_compare = st.selectbox("Année de comparaison", list(range(2012,2021))[::-1], index=1)

        if nats_sel:
            df_sel = df_nat[df_nat['nationalite'].isin(nats_sel)].copy()
            df_melt2 = df_sel.melt(id_vars='nationalite', value_vars=list(range(2012,2021)),
                                   var_name='Année', value_name='Arrivées')
            df_melt2['Année'] = df_melt2['Année'].astype(str)

            fig_l = px.line(df_melt2, x='Année', y='Arrivées', color='nationalite',
                            color_discrete_sequence=COLORS, template='simple_white',
                            markers=True, labels={'nationalite':'Nationalité'})
            fig_l.update_traces(line_width=2.5, marker_size=6)
            fig_l.update_layout(height=340, margin=dict(l=0,r=0,t=10,b=0),
                                 yaxis_tickformat='.2s', paper_bgcolor='white',
                                 plot_bgcolor='white', font_family='Inter',
                                 legend=dict(orientation='h', y=-0.2))
            fig_l.update_yaxes(gridcolor='#F0EBE3')
            fig_l.update_xaxes(showgrid=False)
            st.plotly_chart(fig_l, use_container_width=True)

            # Table comparaison
            st.markdown(f"##### Tableau — Arrivées par nationalité ({yr_compare})")
            df_table = df_nat[df_nat['nationalite'].isin(nats_sel)][['nationalite', yr_compare]].copy()
            df_table.columns = ['Nationalité', f'Arrivées {yr_compare}']
            df_table = df_table.sort_values(f'Arrivées {yr_compare}', ascending=False)
            df_table[f'Arrivées {yr_compare}'] = df_table[f'Arrivées {yr_compare}'].apply(lambda x: f"{x:,.0f}")
            st.dataframe(df_table, use_container_width=True, hide_index=True)

    # ── TAB 2 : PROVINCES ──
    with tab2:
        prov_sel = st.multiselect(
            "Postes à afficher",
            options=df_prov['provinces'].tolist(),
            default=['A Marrakech Ménara', 'A Mohammed V', 'A Tanger Ibn Battouta', 'P Tanger Med', 'T Bab Sebta']
        )
        chart_type = st.radio("Type de graphique", ["Lignes", "Barres groupées", "Aire"], horizontal=True)

        if prov_sel:
            df_psel = df_prov[df_prov['provinces'].isin(prov_sel)].copy()
            df_pmelt = df_psel.melt(id_vars='provinces', value_vars=list(range(2012,2021)),
                                    var_name='Année', value_name='Flux')
            df_pmelt['Année'] = df_pmelt['Année'].astype(str)

            if chart_type == "Lignes":
                figp = px.line(df_pmelt, x='Année', y='Flux', color='provinces',
                               color_discrete_sequence=COLORS, template='simple_white',
                               markers=True)
            elif chart_type == "Barres groupées":
                figp = px.bar(df_pmelt, x='Année', y='Flux', color='provinces',
                              color_discrete_sequence=COLORS, template='simple_white', barmode='group')
            else:
                figp = px.area(df_pmelt, x='Année', y='Flux', color='provinces',
                               color_discrete_sequence=COLORS, template='simple_white')

            figp.update_layout(height=360, margin=dict(l=0,r=0,t=10,b=0),
                                yaxis_tickformat='.2s', paper_bgcolor='white',
                                plot_bgcolor='white', font_family='Inter',
                                legend=dict(orientation='h', y=-0.25))
            figp.update_yaxes(gridcolor='#F0EBE3')
            figp.update_xaxes(showgrid=False)
            st.plotly_chart(figp, use_container_width=True)

        # Heatmap
        st.markdown("##### Carte thermique — Flux par province et par année")
        df_heat = df_prov.set_index('provinces')
        df_heat.columns = [str(c) for c in df_heat.columns]
        fig_h = px.imshow(df_heat, color_continuous_scale=['#EFF6FF','#2A5C8A','#C4562A'],
                          aspect='auto', text_auto='.2s',
                          labels=dict(color="Flux"))
        fig_h.update_layout(height=520, margin=dict(l=0,r=0,t=10,b=0),
                             paper_bgcolor='white', font_family='Inter', font_size=10,
                             coloraxis_showscale=True)
        st.plotly_chart(fig_h, use_container_width=True)

    # ── TAB 3 : NUITÉES ──
    with tab3:
        col_n1, col_n2 = st.columns([2, 1])
        with col_n1:
            df_nmelt = df_nuit.melt(id_vars='provinces', value_vars=list(range(2012,2021)),
                                    var_name='Année', value_name='Nuitées')
            df_nmelt['Année'] = df_nmelt['Année'].astype(str)
            fig_n = px.bar(df_nmelt, x='Année', y='Nuitées', color='provinces',
                           color_discrete_sequence=COLORS, template='simple_white',
                           labels={'provinces':'Destination'})
            fig_n.update_layout(height=360, margin=dict(l=0,r=0,t=10,b=0),
                                 yaxis_tickformat='.2s', paper_bgcolor='white',
                                 plot_bgcolor='white', font_family='Inter',
                                 legend=dict(orientation='h', y=-0.3, font_size=10))
            fig_n.update_xaxes(showgrid=False)
            fig_n.update_yaxes(gridcolor='#F0EBE3')
            st.plotly_chart(fig_n, use_container_width=True)

        with col_n2:
            yr_nuit = st.selectbox("Année", list(range(2012,2021))[::-1], key="yr_nuit")
            df_nyr = df_nuit[['provinces', yr_nuit]].copy().sort_values(yr_nuit, ascending=False)
            fig_np = px.pie(df_nyr, names='provinces', values=yr_nuit, hole=0.5,
                            color_discrete_sequence=COLORS, template='simple_white')
            fig_np.update_layout(height=360, margin=dict(l=0,r=0,t=10,b=0),
                                  paper_bgcolor='white', font_family='Inter',
                                  legend=dict(font_size=10))
            fig_np.update_traces(textinfo='percent', textfont_size=10)
            st.plotly_chart(fig_np, use_container_width=True)

        # Impact COVID comparaison
        st.markdown("##### Impact COVID-19 — 2019 vs 2020")
        df_covid = df_nuit[['provinces', 2019, 2020]].copy()
        df_covid['Chute (%)'] = ((df_covid[2020] - df_covid[2019]) / df_covid[2019] * 100).round(1)
        fig_covid = go.Figure()
        fig_covid.add_trace(go.Bar(name='2019', x=df_covid['provinces'], y=df_covid[2019],
                                   marker_color='#2A5C8A'))
        fig_covid.add_trace(go.Bar(name='2020', x=df_covid['provinces'], y=df_covid[2020],
                                   marker_color='#C4562A'))
        fig_covid.update_layout(barmode='group', height=320, template='simple_white',
                                 margin=dict(l=0,r=0,t=10,b=0),
                                 yaxis_tickformat='.2s', paper_bgcolor='white',
                                 font_family='Inter',
                                 legend=dict(orientation='h', y=1.1))
        fig_covid.update_xaxes(tickangle=30)
        fig_covid.update_yaxes(gridcolor='#F0EBE3')
        st.plotly_chart(fig_covid, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MÉTHODOLOGIE
# ══════════════════════════════════════════════════════════════════════════════
elif "Méthodologie" in page:
    st.markdown("""
    <h1 style='margin-bottom:.25rem'>⚙ Méthodologie</h1>
    <p style='color:#888;margin-bottom:2rem'>Pipeline de Machine Learning — Classification Bayésienne</p>
    """, unsafe_allow_html=True)

    # Pipeline visuel
    steps = [
        ("01", "Collecte", "3 fichiers ONMT fusionnés via pandas.merge() sur l'année", "📥"),
        ("02", "Nettoyage", "Suppression des NaN, harmonisation des colonnes, melt() vers format long", "🧹"),
        ("03", "Feature Eng.", "LabelEncoder (nat/prov), saisonnalité (trimestre), interaction nat × province", "🔧"),
        ("04", "Modélisation", "MultinomialNB — P(Flux Élevé | features) via théorème de Bayes", "🤖"),
        ("05", "Évaluation", "Accuracy ~85%, rapport classification, matrice de confusion, split 80/20", "📊"),
        ("06", "Déploiement", "Sérialisation joblib → .pkl pour réutilisation sans ré-entraînement", "🚀"),
    ]

    cols = st.columns(3)
    for i, (num, title, desc, icon) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="siaha-card" style='min-height:140px'>
              <div style='font-size:.65rem;font-weight:800;color:#C4562A;letter-spacing:.12em;margin-bottom:.5rem'>{num} — {icon}</div>
              <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;margin-bottom:.5rem'>{title}</div>
              <div style='font-size:.82rem;color:#666;line-height:1.55'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("#### Variables du modèle")
        df_vars = pd.DataFrame({
            'Variable': ['nat_encoded', 'prov_encoded', 'trim_encoded', 'inter_encoded', 'target_flux'],
            'Type': ['Feature', 'Feature', 'Feature', 'Feature', 'Cible (Y)'],
            'Description': [
                'Nationalité encodée (LabelEncoder)',
                'Poste-frontière encodé (LabelEncoder)',
                'Trimestre 1–4 (saisonnalité)',
                'Interaction Nationalité × Province',
                '1 si flux > médiane, 0 sinon'
            ]
        })
        st.dataframe(df_vars, use_container_width=True, hide_index=True)

    with col_m2:
        st.markdown("#### Métriques de performance")
        metrics_data = {
            'Métrique': ['Accuracy globale', 'Précision (classe 1)', 'Rappel (classe 1)', 'F1-Score', 'Test set'],
            'Valeur': ['~85%', '~84%', '~86%', '~85%', '20%']
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

    st.markdown("#### Théorème de Bayes appliqué")
    st.markdown(r"""
    Le modèle **Naive Bayes Multinomial** estime :
    $$P(\text{Flux Élevé} \mid \text{Nat}, \text{Province}, \text{Trimestre}) = \frac{P(\text{Nat}, \text{Province}, \text{Trimestre} \mid \text{Élevé}) \cdot P(\text{Élevé})}{P(\text{Nat}, \text{Province}, \text{Trimestre})}$$

    L'hypothèse d'**indépendance conditionnelle** (Naive) simplifie le calcul :
    $$P(x_1, x_2, x_3, x_4 \mid C) = \prod_{i=1}^{4} P(x_i \mid C)$$
    """)

    st.markdown("---")
    st.markdown("#### Classes disponibles dans les encodeurs")
    with st.expander("Voir toutes les classes"):
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Nationalités**")
        for n in le_nat.classes_:
            c1.markdown(f"- {NAT_FLAGS.get(n,'')} {n}")
        c2.markdown("**Postes-frontières**")
        for p in le_prov.classes_:
            c2.markdown(f"- {POSTE_LABELS.get(p, p)}")
        c3.markdown("**Trimestres**")
        for t in le_trim.classes_:
            c3.markdown(f"- T{t} : {TRIM_LABELS[t]}")
