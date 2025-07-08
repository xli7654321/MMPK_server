import streamlit as st

st.set_page_config(
    page_title='Home',
    layout='wide'
)

st.header('MMPK')
st.markdown(
    """
    <div style='font-size:20px; margin-bottom:24px;'>
        MMPK is an end-to-end multimodal deep learning model for 
        <span style='color:#FF4B4B; font-weight:bold;'>
        human oral pharmacokinetic parameters</span> prediction. 
        It leverages multiple molecular representations learned from molecular graphs, substructure graphs, and SMILES sequences to comprehensively capture multi-scale molecular information.
        Multi-task learning and data imputation strategies are also employed to improve data efficiency and model robustness.
        Additionally, the substructure-aware cross-attention mechanism enhances the interpretability of MMPK by identifying chemically meaningful substructures that contribute to the predictions.
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 4, 1])
with open('mmpk.svg', 'r', encoding='utf-8') as f:
    mmpk_svg = f.read()
with col2:
    st.image(mmpk_svg, width=1000)

col4, col5 = st.columns([1, 1])
with col4:
    st.subheader('Pharmacokinetic Parameters', divider='gray')
    st.markdown(
    """
    <div style='font-size:20px;'>
        MMPK can predict the following eight human oral pharmacokinetic parameters:
        <ul>
            <li>Area under the concentration–time curve (<span style='color:#FF4B4B; font-weight:bold;'>AUC</span>)</li>
            <li>Maximum plasma concentration (<span style='color:#FF4B4B; font-weight:bold;'>C<sub>max</sub></span>)</li>
            <li>Time to reach C<sub>max</sub> (<span style='color:#FF4B4B; font-weight:bold;'>T<sub>max</sub></span>)</li>
            <li>Elimination half-life (<span style='color:#FF4B4B; font-weight:bold;'>t<sub>1/2</sub></span>)</li>
            <li>Apparent clearance (<span style='color:#FF4B4B; font-weight:bold;'>CL/F</span>)</li>
            <li>Apparent volume of distribution (<span style='color:#FF4B4B; font-weight:bold;'>V<sub>z</sub>/F</span>)</li>
            <li>Mean residence time (<span style='color:#FF4B4B; font-weight:bold;'>MRT</span>)</li>
            <li>Oral absolute bioavailability (<span style='color:#FF4B4B; font-weight:bold;'>F</span>)</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
with col5:
    st.subheader('Contact', divider='gray')
    st.markdown('#### Xiang Li')
    st.markdown(
        """
        <div style='font-size:20px; margin-bottom:12px;'>
        📧 E-mail:
        <a href="mailto:xli3667@mail.ecust.edu.cn" style="color:#FF4B4B;">xli3667&#8203;@mail.ecust.edu.cn</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    # badge1 = '[![X Follow](https://img.shields.io/twitter/follow/xli7654321)](https://x.com/xli7654321)'
    # badge2 = '[![GitHub](https://img.shields.io/static/v1?label=GitHub&message=@xli7654321&color=9e51eb&logo=github)](https://github.com/xli7654321)'
    
    badge1 = '<a href="https://x.com/xli7654321" target="_blank"><img style="height:24px;" src="https://img.shields.io/twitter/follow/xli7654321" alt="X Follow"></a>'
    badge2 = '<a href="https://github.com/xli7654321" target="_blank"><img style="height:24px;" src="https://img.shields.io/static/v1?label=GitHub&message=xli7654321&color=9e51eb&logo=github" alt="GitHub"></a>'
    st.markdown(
        f"{badge1}&nbsp;&nbsp;&nbsp;{badge2}",
        unsafe_allow_html=True
    )
    
    st.markdown('#### Weihua Li')
    st.markdown(
        """
        <div style='font-size:20px; margin-bottom:18px;'>
        📧 E-mail:
        <a href="mailto:whli@ecust.edu.cn" style="color:#FF4B4B;">whli&#8203;@ecust.edu.cn</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('#### Our Lab')
    st.markdown(
        """
        <div style='font-size:20px;'>
        🔗
        <a href="https://lmmd.ecust.edu.cn/" target="_blank" 
           style="color:#FF4B4B;" id="lmmd_link">Laboratory of Molecular Modeling and Design</a>
        </div>
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

# 2. documentation

