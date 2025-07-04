import numpy as np
from scipy.optimize import fsolve
import plotly.graph_objects as go
import plotly.colors as pc

def simulate_pk_curve(
    dose: float,
    dose_unit: str,
    auc: float,
    cmax: float,
    tmax: float,
    t_half: float,
    cl_over_f: float,
    vz_over_f: float,
    t_end: float = 24,
    n_points: int = 241,
    verbose: bool = True
):
    # 1. ke
    ke = np.log(2) / t_half
    
    # 2. Consistency check
    if dose_unit == 'mg':
        dose = dose * 1000  # mg -> ug
    elif dose_unit == 'mg/kg':
        dose = dose * 70 * 1000  # mg/kg -> ug
    
    err_cl = abs(dose / auc - cl_over_f) / cl_over_f * 100
    err_vz  = abs(cl_over_f / ke - vz_over_f) / vz_over_f * 100
    if verbose:
        if err_cl > 5:
            print(f"[Warning] Predicted CL/F ({cl_over_f:.3g}) inconsistent with Dose/AUC ({dose / auc:.3g}) (deviation {err_cl:.1f} %)")
        if err_vz  > 5:
            print(f"[Warning] Predicted Vz/F ({vz_over_f:.3g}) inconsistent with CL/F/ke ({cl_over_f / ke:.3g}) (deviation {err_vz:.1f} %)")
    
    # 3. ka
    def solve_ka(ka):
        return (np.log(ka) - np.log(ke)) / (ka - ke) - tmax
    
    # Initial guess, slightly larger than ke (ka > ke is typical for oral PK)
    ka_init = ke + 0.1
    ka = fsolve(solve_ka, x0=ka_init)[0]
    
    coeff = dose * ka / (vz_over_f * (ka - ke))
    cmax_simulated = coeff * (np.exp(-ke * tmax) - np.exp(-ka * tmax))
    scale = cmax / cmax_simulated
    coeff *= scale
    
    t = np.linspace(0, t_end, n_points)
    C_t = coeff * (np.exp(-ke * t) - np.exp(-ka * t))
    C_t[C_t < 0] = 0
    
    return t, C_t, ka

colors = pc.qualitative.Plotly

def plotly_pk_curve(curves):
    fig = go.Figure()

    for i, curve in enumerate(curves):
        t, C_t = curve['t'], curve['C_t']
        tmax, cmax = curve['tmax'], curve['cmax']
        ka, t_half = curve['ka'], curve['t_half']
        dose = curve['dose']
        dose_unit = curve['dose_unit']
        color = colors[i % len(colors)]

        # curve
        fig.add_trace(go.Scatter(
            x=t,
            y=C_t,
            mode='lines',
            name=f"Dose: {dose} {dose_unit}",
            line=dict(width=2, color=color),
            hovertemplate='Conc: %{y:.2f} ng/mL<br>Time: %{x:.1f} h',
            # fill='tozeroy'
        ))

        # cmax
        fig.add_trace(go.Scatter(
            x=[tmax],
            y=[cmax],
            mode='markers',
            name=f"Dose: {dose} {dose_unit}",
            marker=dict(color='red', size=10, symbol='circle'),
            text=[f"Cmax = {cmax:.2f} ng/mL<br>Tmax = {tmax:.1f} h"],
            hoverinfo='text',
            showlegend=False
        ))
        
        # ka
        t_half_a = np.log(2) / ka
        c_half_a = np.interp(t_half_a, t, C_t)
        fig.add_trace(go.Scatter(
            x=[t_half_a],
            y=[c_half_a],
            mode='markers',
            name=f"Dose: {dose} {dose_unit}",
            marker=dict(color='red', size=10, symbol='diamond'),
            text=[f"Conc = {c_half_a:.2f} ng/mL<br>t_half_a = {t_half_a:.1f} h"],
            hoverinfo='text',
            showlegend=False
        ))
        
        # t_half
        t_half_e = t_half
        c_half_e = np.interp(t_half_e, t, C_t)
        fig.add_trace(go.Scatter(
            x=[t_half_e],
            y=[c_half_e],
            mode='markers',
            marker=dict(color='red', size=10, symbol='star'),
            text=[f"Conc = {c_half_e:.2f} ng/mL<br>t_half_e = {t_half_e:.1f} h"],
            hoverinfo='text',
            showlegend=False
        ))
        
    fig.update_layout(
        xaxis_title='Time (h)',
        yaxis_title='Concentration (ng/mL)',
        template='plotly_white',
        height=500
    )
    return fig

if __name__ == '__main__':
    pass
