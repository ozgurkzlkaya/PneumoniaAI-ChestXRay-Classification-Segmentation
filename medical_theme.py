import streamlit as st


def apply_medical_theme():
    st.markdown('''<style>
    :root {--med-teal:#0E938F;--med-navy:#173B46;--med-border:#DCEBEA}
    .stApp {background:radial-gradient(circle at 95% 2%,#E0F6F4 0%,transparent 32%),#F5FAFA;color:#173B46}
    [data-testid="stSidebar"] {background:linear-gradient(180deg,#FFFFFF,#F1FAF9);border-right:1px solid #DCEBEA}
    [data-testid="stSidebarNav"] {display:none!important}
    [data-testid="stSidebar"] [data-testid="stPageLink"] {border-radius:10px;margin:3px 0}
    [data-testid="stSidebar"] [data-testid="stPageLink"]:hover {background:#DDF4F0}
    .med-sidebar-brand {display:flex;align-items:center;gap:12px;padding:16px 4px 18px}
    .med-sidebar-icon {width:49px;height:49px;border-radius:12px;background:linear-gradient(140deg,#087E84,#1DB4AC);display:flex;align-items:center;justify-content:center;font-size:29px;color:white}
    .med-sidebar-name {font-size:19px;font-weight:800;color:#173B46}
    .med-sidebar-tag {font-size:10px;letter-spacing:.16em;font-weight:700;color:#0E938F}
    .med-sidebar-label {font-size:10px;font-weight:800;letter-spacing:.16em;color:#7A989C;margin:8px 4px 12px}
    .med-sidebar-footer {border-top:1px solid #DCEBEA;margin-top:34px;padding:18px 5px;color:#739092;font-size:12px;line-height:1.7}
    .med-sidebar-footer strong {font-size:10px;letter-spacing:.1em}
    .block-container {max-width:1320px;padding-top:2rem}
    [data-testid="stMetric"] {background:linear-gradient(145deg,#fff,#EAF8F6);border:1px solid #D5EBE8;border-radius:15px;padding:18px}
    [data-testid="stMetricValue"] {color:#087D79}
    div[data-baseweb="tab-list"] {background:#fff;border:1px solid #DCEBEA;border-radius:14px;padding:7px;gap:6px}
    button[data-baseweb="tab"] {border-radius:10px;font-weight:600}
    button[data-baseweb="tab"][aria-selected="true"] {background:#DDF4F0;color:#087D79}
    [data-testid="stDataFrame"] {border:1px solid #DCEBEA;border-radius:12px;overflow:hidden}
    .med-section {background:linear-gradient(110deg,#0C7779,#15A6A1);color:white;border-radius:17px;padding:24px 28px;margin:8px 0 22px}
    .med-section h2 {color:white!important;margin:0 0 8px}
    .med-section p {color:#D9FAF5;margin:0}
    </style>''', unsafe_allow_html=True)


def render_medical_sidebar():
    with st.sidebar:
        st.markdown('''<div class="med-sidebar-brand">
          <div class="med-sidebar-icon">&#129729;</div>
          <div><div class="med-sidebar-name">Pneumonia AI</div>
          <div class="med-sidebar-tag">TIBBİ GÖRÜNTÜLEME ARAŞTIRMASI</div></div></div>
          <div class="med-sidebar-label">SAYFALAR</div>''', unsafe_allow_html=True)
        st.page_link('app.py', label='Görüntü Analizi', icon=':material/home:', use_container_width=True)
        st.page_link('pages/1_Hata_Analizi.py', label='Hata Analizi', icon=':material/analytics:', use_container_width=True)
        st.page_link('pages/2_Platform_Rehberi.py', label='Proje Hakkında', icon=':material/menu_book:', use_container_width=True)
        st.markdown('''<div class="med-sidebar-footer"><strong>ARAŞTIRMA SÜRÜMÜ</strong>
          <p>Akademik araştırma prototipi.<br>Klinik tanı aracı değildir.</p></div>''', unsafe_allow_html=True)
