import streamlit as st
from streamlit_ketcher import st_ketcher
from streamlit.components.v1 import html as st_html
import re
import html
import datetime
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import torch
from rdkit import Chem
from config import args
from dataloader import MMPKPredictLoader
from mmpk import MMPKPredictor
from utils import back_transform_predict, seed_everything, standardize
from predict import test, TASKS
from attention import show_mol_svg, show_cbar_svg, show_sub_svg, svg_to_data_uri
from curve import simulate_pk_curve, plotly_pk_curve
from streamlit.web.server.websocket_headers import _get_websocket_headers

def load_single_example():
    st.session_state['single_smi_input'] = 'C[C@]12C[C@H](c3ccc(S(C)(=O)=O)cc3)C3=C4CCC(=O)C=C4CC[C@H]3[C@@H]1CC[C@@]2(O)C(F)(F)C(F)(F)F'
    st.session_state['single_dose_input'] = 2.00
def reset_single_inputs():
    st.session_state['single_smi_input'] = None
    st.session_state['single_dose_input'] = None

def load_batch_example():
    st.session_state['batch_smi_input'] = """\
C[C@]12C[C@H](c3ccc(S(C)(=O)=O)cc3)C3=C4CCC(=O)C=C4CC[C@H]3[C@@H]1CC[C@@]2(O)C(F)(F)C(F)(F)F
C[C@]12C[C@H](c3ccc(S(C)(=O)=O)cc3)C3=C4CCC(=O)C=C4CC[C@H]3[C@@H]1CC[C@@]2(O)C(F)(F)C(F)(F)F
"""
    st.session_state['batch_dose_input'] = '2.00\n4.00'
def reset_batch_inputs():
    st.session_state['batch_smi_input'] = None
    st.session_state['batch_dose_input'] = None

def is_valid_smiles(smi: str) -> bool:
    return Chem.MolFromSmiles(smi) is not None

@st.cache_resource(show_spinner=False)
def load_model(fold):
    model = MMPKPredictor(args, num_tasks=num_tasks)
    model_path = f"checkpoints/{args.checkpoints_folder}/fold_{fold}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

def round_value(values, digits=2):
    fmt = '0.' + '0' * digits
    return [float(Decimal(str(v)).quantize(Decimal(fmt), rounding=ROUND_HALF_UP)) for v in values]

@st.fragment
def download_csv_btn(label: str, csv: bytes, file_name: str, mime='text/csv'):
    st.download_button(
        label=label,
        data=csv,
        file_name=file_name,
        mime=mime,
    )

def log_submit_event():
    with open('submit_log.txt', 'a') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[Submit] {timestamp}\n")

st.set_page_config(
    page_title='PK Prediction',
    layout='wide'
)

st.header('MMPK')

input_mode = st.selectbox(
    '**Select a Method to Enter Molecules:**',
    ['Draw Molecule', 'Single SMILES', 'Batch SMILES', 'Upload File']
)

if input_mode == 'Single SMILES':
    smi = st.text_input(
        'Enter a SMILES string:',
        key='single_smi_input'
    )
    dose = st.number_input(
        'Enter the administered dose:',
        min_value=0.0,
        value=None,
        step=0.01,
        format='%0.2f',
        key='single_dose_input'
    )
    
    smi_list, doses = [], []
    if smi and dose:
        smi_list = [smi]
        doses = [dose]

    col_1, col_2, spacer, col3 = st.columns(
        [0.2, 0.2, 0.45, 0.2],
        gap='medium',
        vertical_alignment='bottom'
    )
    with col_1:
        dose_unit = st.selectbox(
            'Select dose unit:',
            ('mg', 'mg/kg')
        )
    with col_2:
        standardize_smi = st.checkbox(
            'Standardize SMILES',
            value=True
        )
    with col3:
        btn1, btn2 = st.columns([1.5, 1], gap='small', vertical_alignment='bottom')
        with btn1:
            st.button('Example', on_click=load_single_example)
        with btn2:
            st.button('Reset', on_click=reset_single_inputs)
elif input_mode == 'Draw Molecule':
    smi = st_ketcher()
    st_html(
        """
        <script>
        window.onload = function () {
            const iframe = window.parent.document.getElementsByClassName('stCustomComponentV1')[0];
            if (!iframe) return;

            iframe.onload = function () {
                const innerDoc = iframe.contentDocument || iframe.contentWindow.document;
                const style = innerDoc.createElement('style');
                style.innerHTML = `
                    @font-face {
                        font-family: 'MyFont';
                        src: url('static/ProximaSoft-Regular.woff2') format('woff2');
                        font-weight: 400;
                        font-style: normal;
                    }
                    button.css-b0hyfk {
                        font-family: 'MyFont', sans-serif !important;
                        font-size: 18px !important;
                    }
                `;
                innerDoc.head.appendChild(style);
                const buttons = innerDoc.querySelectorAll('button.css-b0hyfk');
                buttons.forEach(btn => {
                    btn.style.fontFamily = 'MyFont, sans-serif';
                    btn.style.fontSize = '18px';
                });
            };
        };
        </script>
        """,
        height=0
    )
    st.info('👉 Please click the **Apply** button at the top-right to apply your drawn molecule.')
    if smi:
        smi = standardize(smi)
        st.success(f"✅ Applied SMILES: **{smi}**")
    dose = st.number_input(
        'Enter the administered dose:',
        min_value=0.0,
        value=None,
        step=0.01,
        format='%0.2f'
    )
    smi_list, doses = [], []
    if smi and dose:
        smi_list = [smi]
        doses = [dose]
    col_1, col_2 = st.columns(
        [0.2, 0.8],
        gap='medium',
        vertical_alignment='bottom'
    )
    with col_1:
        dose_unit = st.selectbox(
            'Select dose unit:',
            ('mg', 'mg/kg')
        )
    with col_2:
        st.empty()
    standardize_smi = True
elif input_mode == 'Batch SMILES':
    smi_text = st.text_area(
        'Enter multiple SMILES (one per line or separated by commas):',
        key='batch_smi_input')
    dose_text = st.text_area(
        'Enter the administered dose (one per line or separated by commas):',
        key='batch_dose_input')
    
    smi_list, doses = [], []
    if smi_text and dose_text:
        smi_list = [s.strip() for s in re.split(r'[\n,]+', smi_text) if s.strip()]
        doses_str = [s.strip() for s in re.split(r'[\n,]+', dose_text) if s.strip()]
        doses = [float(d) for d in doses_str]

        if len(smi_list) > 100 or len(doses) > 100:
            st.warning('⚠️ A maximum of 100 SMILES–dose combinations is allowed. Please reduce your input.')
            smi_list, doses = [], []
    
    col_1, col_2, spacer, col3 = st.columns(
        [0.2, 0.2, 0.45, 0.2],
        gap='medium',
        vertical_alignment='bottom'
    )
    with col_1:
        dose_unit = st.selectbox(
            'Select dose unit:',
            ('mg', 'mg/kg')
        )
    with col_2:
        standardize_smi = st.checkbox(
            'Standardize SMILES',
            value=True
        )
    with col3:
        btn1, btn2 = st.columns([1.5, 1], gap='small', vertical_alignment='bottom')
        with btn1:
            st.button('Example', on_click=load_batch_example)
        with btn2:
            st.button('Reset', on_click=reset_batch_inputs)
elif input_mode == 'Upload File':
    csv_file = st.file_uploader(
        "Upload a **CSV** file with `smiles` and `dose` columns:",
        type=['csv'],
        key='file_input'
    )
    if csv_file:
        df = pd.read_csv(csv_file)
        if 'smiles' not in df.columns or 'dose' not in df.columns:
            st.error("❌ The uploaded file must contain 'smiles' and 'dose' columns.")
        else:
            df = df[['smiles', 'dose']].dropna()
            
            if df.empty:
                st.error("❌ The uploaded file must contain at least one row with 'smiles' and 'dose' values.")
            elif len(df) > 500:
                st.warning('⚠️ A maximum of 500 SMILES–dose combinations is allowed per file. Please re-upload the file.')
            else:
                smi_list = df['smiles'].astype(str).tolist()
                doses = df['dose'].astype(float).tolist()
                st.success(f"✅ Successfully loaded **{len(smi_list)}** SMILES–dose combinations.")
    
    col_1, col_2, spacer, col3 = st.columns(
        [0.2, 0.2, 0.45, 0.2],
        gap='medium',
        vertical_alignment='bottom'
    )
    with col_1:
        dose_unit = st.selectbox(
            'Select dose unit:',
            ('mg', 'mg/kg')
        )
    with col_2:
        standardize_smi = st.checkbox(
            'Standardize SMILES',
            value=True
        )
    with col3:
        with open('examples/example.csv', 'rb') as f:
            eg_csv = f.read() 
        download_csv_btn(
            label='⬇️ Download Example',
            csv=eg_csv,
            file_name='example.csv'
        )

st.markdown("""<div style='margin-top: 30px;'></div>""", unsafe_allow_html=True)

if st.button('Submit', type='primary'):
    log_submit_event()
    if 'smi_list' not in locals() or 'doses' not in locals():
        st.error('❌ Please provide input or upload a file.')
    elif not smi_list or not doses or len(smi_list) != len(doses):
        st.error('❌ The number of SMILES and doses must match and cannot be empty!')
    else:
        invalid_smis = [s for s in smi_list if not is_valid_smiles(s)]
        if invalid_smis:
            st.error(f"❌ The following SMILES are invalid: {', '.join(invalid_smis)}")
        else:
            with st.spinner('Predicting, please wait...', show_time=True):
                seed_everything(args.seed)
                device = torch.device(
                    'cuda:' + str(args.device)
                    if torch.cuda.is_available() else 'cpu'
                )
                num_tasks = len(args.pk_params)
                df_preds_by_task = {task: [] for task in TASKS}
                att_pred = []
                for fold in range(1, 11):
                    model = load_model(fold)
                    loader = MMPKPredictLoader(smi_list, doses, dose_unit, standardize_smi).get_loader()
                    preds, att_info = test(model, loader, device)
                    for task in TASKS:
                        df_preds = pd.DataFrame(preds[task])
                        df_preds_by_task[task].append(df_preds)
                    for info in att_info:
                        for sub in info['sub_scores']:
                            att_pred.append({
                                'fold': fold,
                                'smiles': info['smiles'],
                                'dose': info['dose'],
                                'sub_index': sub['sub_index'] + 1,
                                'sub_smiles': sub['sub_smiles'],
                                'sub_score': sub['sub_score']
                            })
                ref_df = df_preds_by_task[TASKS[0]][0][['smiles', 'dose']].reset_index(drop=True)
                pred_df = ref_df.copy()
                for task in TASKS:
                    dfs = df_preds_by_task[task]
                    merged_df = pd.concat([
                        df[['y_hat']].rename(columns={'y_hat': f"y_hat_fold_{i+1}"})
                        for i, df in enumerate(dfs)
                    ], axis=1)
                    pred_df[f"{task}_log"] = merged_df.mean(axis=1)
                    y_hat_orig = back_transform_predict(y_hat_log=pred_df[f"{task}_log"], pk_param=task)
                    pred_df[task] = round_value(y_hat_orig, digits=2)
                task_cols = TASKS
                log_cols = [f"{task}_log" for task in TASKS]
                ordered_cols = ['smiles', 'dose'] + task_cols + log_cols
                pred_df = pred_df[ordered_cols]
                unit_mapping = {
                    'smiles': 'SMILES',
                    'dose': f"Dose [{dose_unit}]",
                    'auc': 'AUC [ng*h/mL]',
                    'cmax': 'Cmax [ng/mL]',
                    'tmax': 'Tmax [h]',
                    'hl': 't1/2 [h]',
                    'cl': 'CL/F [L/h]',
                    'vz': 'Vz/F [L]',
                    'mrt': 'MRT [h]',
                    'f': 'F [%]',
                    'auc_log': 'AUC [log]',
                    'cmax_log': 'Cmax [log]',
                    'tmax_log': 'Tmax [log]',
                    'hl_log': 't1/2 [log]',
                    'cl_log': 'CL/F [log]',
                    'vz_log': 'Vz/F [log]',
                    'mrt_log': 'MRT [log]',
                    'f_log': 'F [logit]'
                }
                pred_df = pred_df.rename(columns=unit_mapping)
                cols_to_display = [col for col in pred_df.columns if not col.endswith(('[log]', '[logit]'))]
                pred_df_display = pred_df[cols_to_display]
                pred_df_display.index = pred_df_display.index + 1
                pred_df_display.index.name = 'No.'
                
                df_att = pd.DataFrame(att_pred)
                df_att_mean = (
                    df_att
                    .groupby(['smiles', 'sub_smiles'], as_index=False)
                    ['sub_score']
                    .mean()
                )
                df_att_mean['sub_score'] = round_value(df_att_mean['sub_score'], digits=3)
                att_col_mapping = {
                    'smiles': 'SMILES',
                    'sub_smiles': 'Substructure SMILES',
                    'sub_score': 'Attention Weight'
                }
                df_att_mean = df_att_mean.rename(columns=att_col_mapping)
                df_att_mean.index = df_att_mean.index + 1
                df_att_mean.index.name = 'No.'
                
                df_att_mean['SMILES'] = pd.Categorical(
                    df_att_mean['SMILES'],
                    categories=df_att['smiles'].drop_duplicates().tolist(),
                    ordered=True
                )
                df_att_mean = df_att_mean.sort_values('SMILES').reset_index(drop=True)
                
                st.success('💡 Prediction completed!')

                tab1, tab2, tab3 = st.tabs(['Predicted PK Parameters', 'Attention Visualization', 'Simulated PK Curve'])

                with tab1:
                    pred_df_display = pred_df_display.copy()
                    pred_df_display['Molecular Image'] = pred_df_display['SMILES'].apply(show_sub_svg).apply(svg_to_data_uri)
                    cols = pred_df_display.columns.tolist()
                    cols = ['Molecular Image'] + [col for col in cols if col != 'Molecular Image']
                    pred_df_display = pred_df_display[cols]
                    st.dataframe(
                        pred_df_display,
                        column_config={
                            'Molecular Image': st.column_config.ImageColumn(
                                'Molecular Image', width='medium'
                            )
                        }
                    )
                    
                    download_df = pred_df_display.drop(columns=['Molecular Image'])
                    download_csv = download_df.to_csv(index=False).encode('utf-8')
                    download_csv_btn(
                        label='⬇️ Download Prediction Results',
                        csv=download_csv,
                        file_name='human_oral_pk_predictions.csv'
                    )
                with tab2:
                    st.info('📌 **Note:** Here, the substructure attention weights are **independent of dose**.')
                    smi_list_uni = df_att_mean['SMILES'].unique()
                    for i, smi in enumerate(smi_list_uni):
                        sub_df = df_att_mean[df_att_mean['SMILES'] == smi].reset_index(drop=True)
                        max_row = sub_df.loc[sub_df['Attention Weight'].idxmax()]
                        sub_df['Substructure Image'] = sub_df['Substructure SMILES'].apply(show_sub_svg).apply(svg_to_data_uri)
                        sub_df.index = sub_df.index + 1
                        sub_df.index.name = 'No.'
                        
                        mol_svg = show_mol_svg(
                            max_row['SMILES'],
                            max_row['Substructure SMILES'],
                            max_row['Attention Weight'],
                            size=(500, 500)
                        )
                        cbar_svg = show_cbar_svg(figsize=(0.3, 5))

                        safe_smi = html.escape(smi).replace('[', '&#91;').replace(']', '&#93;').replace('(', '&#40;').replace(')', '&#41;')
                        with st.expander(f"**Molecule No.{i+1}:** {safe_smi}"):
                            st.dataframe(
                                sub_df,
                                column_config={
                                    'Substructure Image': st.column_config.ImageColumn(
                                        'Substructure Image', width='medium'
                                    )
                                }
                            )
                            col_m, col_c, spacer = st.columns([5, 1, 12], vertical_alignment='center')
                            with col_m:
                                if mol_svg is not None:
                                    st.image(mol_svg, caption='Top-scoring substructure')
                                else:
                                    st.warning('⚠️ Unable to highlight the top-scoring substructure.')
                            with col_c:
                                st.image(cbar_svg)
                with tab3:
                    st.warning('🚧 **Disclaimer:** The simulated plasma concentration–time curves are for reference only and may not be valid for all compounds, especially those with complex or non-linear PK behavior.')
                    st.info('📌 **Note:** Red diamond, circle, and star markers on the curve indicate the absorption half-life, Tmax, and elimination half-life, respectively.')
                    
                    for i, (smi, group) in enumerate(pred_df.groupby('SMILES', sort=False)):
                        curves = []
                        unique_rows = group.drop_duplicates(subset=[f"Dose [{dose_unit}]"])
                        for _, row in unique_rows.iterrows():
                            pred_params = dict(
                                dose=row[f"Dose [{dose_unit}]"],
                                dose_unit=dose_unit,
                                auc=row['AUC [ng*h/mL]'],
                                cmax=row['Cmax [ng/mL]'],
                                tmax=row['Tmax [h]'],
                                t_half=row['t1/2 [h]'],
                                cl_over_f=row['CL/F [L/h]'],
                                vz_over_f=row['Vz/F [L]']
                            )
                            t, C_t, ka = simulate_pk_curve(**pred_params,
                                                           t_end=96,
                                                           n_points=961,
                                                           verbose=False)
                            curves.append({
                                't': t,
                                'C_t': C_t,
                                'tmax': row['Tmax [h]'],
                                'cmax': row['Cmax [ng/mL]'],
                                'ka': ka,
                                't_half': row['t1/2 [h]'],
                                'dose': row[f"Dose [{dose_unit}]"],
                                'dose_unit': dose_unit
                            })

                        safe_smi = html.escape(smi).replace('[', '&#91;').replace(']', '&#93;').replace('(', '&#40;').replace(')', '&#41;')
                        with st.expander(f"**Molecule No.{i+1}:** {safe_smi}"):
                            fig = plotly_pk_curve(curves)
                            st.plotly_chart(fig, use_container_width=True)

# CSS
st.markdown("""
    <style>
    /* selectbox style */
    div.stSelectbox label div[data-testid="stMarkdownContainer"] > p {
        font-size: 20px;
    }
    div.stTextInput label div[data-testid="stMarkdownContainer"] > p {
        font-size: 18px;
    }
    div.stNumberInput label div[data-testid="stMarkdownContainer"] > p {
        font-size: 18px;
    }
    div.stTextArea label div[data-testid="stMarkdownContainer"] > p {
        font-size: 18px;
    }
    div.stFileUploader label div[data-testid="stMarkdownContainer"] > p {
        font-size: 18px;
    }
    div.stExpander details summary div[data-testid="stMarkdownContainer"] > p {
        font-size: 18px;
    }
    
    button[data-baseweb="tab"] div[data-testid="stMarkdownContainer"] > p {
        font-size: 18px;
    }
    
    div.stColumn div div div > div[data-testid="stButton"] {
        display: flex;
        justify-content: flex-end;
    }
    
    div.stColumn div div div > div[data-testid="stDownloadButton"] {
        display: flex;
        justify-content: flex-end;
    }
    
    .stElementContainer:has(iframe[data-testid="stIFrame"]) {
        display: none !important;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

st.divider()
st.markdown(
    """
    <div style='font-size:18px;'>
    Copyright © 2025 Laboratory of Molecular Modeling and Design, School of Pharmacy, East China University of Science and Technology. All rights reserved.<br>
    Last update: May 10, 2025
    </div>
    """,
    unsafe_allow_html=True
)

# envs/mmpk/Lib/site-packages/streamlit_ketcher/frontend/static/ProximaSoft-Regular.woff2
