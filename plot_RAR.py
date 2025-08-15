# Interactive RAR plot using Plotly.
import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils_analysis.Vobs_fits import Vbar_sq, MOND_vsq

a_0 = 1.2e-10  # MOND acceleration constant in m/s²

def load_sparc_table():
    file = "/mnt/users/koe/SPARC_Lelli2016c.mrt.txt"
    SPARC_c = ["Galaxy", "T", "D", "e_D", "f_D", "Inc",
               "e_Inc", "L", "e_L", "Reff", "SBeff", "Rdisk",
               "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat", "Q", "Ref."]
    return pd.read_fwf(file, skiprows=98, names=SPARC_c)

def get_acceleration(vel, rad):
    """
    Calculate the gravitational acceleration (g_obs or g_bar).
    In the data, vel is in km/s, and rad is in kpc.
    Returns g_obs in m/s².
    """
    return (vel * 1e3)**2 / (rad * 3.086e19)  # Convert to m/s²

def superscript_number(n):
    super_digits = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return str(n).translate(super_digits)

# --- Various families of IFs ---
def n_family(n:float, g_bar):
    return ( 0.5 + np.sqrt( 1 + 4 * ( g_bar / a_0 )**(-n) ) / 2.0 )**( 1/n ) * g_bar

def delta_family(delta:float, g_bar):
    return ( 1 - np.exp( -( g_bar / a_0 )**( delta / 2.0 ) ) )**( -1.0 / delta ) * g_bar

def gamma_family(gamma:float, g_bar):
    y = g_bar / a_0
    return ( ( 1 - np.exp( -y**( gamma / 2.0 ) ) )**( -1.0 / gamma ) + ( 1 - 1.0 / gamma ) * np.exp( -y**( gamma / 2.0 ) ) ) * g_bar

# --- Specific Interpolating Functions (IFs) ---
def rar_if(g_bar):
    # return g_bar / (1 - np.exp(-np.sqrt( g_bar / a_0 )))
    return delta_family( 1.0, g_bar )

def simple_if(g_bar):
    # return g_bar/2 + np.sqrt( g_bar**2 / 4 + g_bar * a_0 )
    return n_family( 1.0, g_bar )

def standard_if(g_bar):
    # return np.sqrt( g_bar**2 / 2.0 + np.sqrt( g_bar**2 * ( g_bar**2 + 4 * a_0**2 ) ) / 2.0 )
    return n_family( 2.0, g_bar )

# --- Load SPARC data ---
sparc_data = {}
table = load_sparc_table()
galaxies = table["Galaxy"].values
galaxy_count = len(galaxies)

columns = [ "Rad", "Vobs", "errV", "Vgas",
            "Vdisk", "Vbul", "SBdisk", "SBbul" ]

for i in tqdm(range(galaxy_count), desc="SPARC galaxies"):
    g = table["Galaxy"][i]
    file_path = f"/mnt/users/koe/data/{g}_rotmod.dat"
    rawdata = np.loadtxt(file_path)
    data = pd.DataFrame(rawdata, columns=columns)

    bulged = np.any(data["Vbul"]>0) # Check whether galaxy has bulge.
    r = data["Rad"]

    Vobs = data["Vobs"]
    Vbar = np.sqrt( Vbar_sq(data, bulged) )
    g_obs = get_acceleration(Vobs, r)
    g_bar = get_acceleration(Vbar, r)

    sparc_data[g] = {
        'r': r,
        'Vobs': Vobs,
        'Vbar': Vbar,
        'g_obs': g_obs,
        'g_bar': g_bar,
        }


# Flatten into background arrays
all_g_obs, all_g_bar, all_galaxy_names = [], [], []
for gal, vals in sparc_data.items():
    all_g_obs.extend(vals['g_obs'])
    all_g_bar.extend(vals['g_bar'])
    all_galaxy_names.extend([gal] * len(vals['g_bar']))  # repeat galaxy name for each point

# Create figure with two subplots
fig = make_subplots(
    rows=2, cols=2,
    specs=[
        [{"rowspan": 2}, {}],  # Row 1: RAR in col 1 (rowspan), Acc curves in col 2
        [None, {}]             # Row 2: empty under RAR, RC in col 2
    ],
    subplot_titles=("RAR", "Rotation Curves", "Acceleration Curves"),
    vertical_spacing=0.15,
    horizontal_spacing=0.15
)

# --- Subplot (1,1): RAR ---
# Background points
fig.add_trace(go.Scatter(
    x=all_g_bar,
    y=all_g_obs,
    mode='markers',
    marker=dict(color='lightgrey', size=5),
    customdata=all_galaxy_names,
    hovertemplate="Galaxy: %{customdata}<br>g_bar: %{x:.2e}<br>g_obs: %{y:.2e}<extra></extra>",
    name="All galaxies",
    legendgroup="rar",
    showlegend=True
), row=1, col=1)

# IF lines
x_line = np.logspace(-13, -8, 100)
fig.add_trace(go.Scatter( x=x_line, y=x_line, mode='lines', name='y = x',
    line=dict(color='black', width=2, dash='dash') ), row=1, col=1)
fig.add_trace(go.Scatter( x=x_line, y=simple_if(x_line), mode='lines', name='Simple IF',
    line=dict(color='orange', width=2) ), row=1, col=1)
fig.add_trace(go.Scatter( x=x_line, y=standard_if(x_line), mode='lines', name='Standard IF',
    line=dict(color='green', width=2) ), row=1, col=1)
fig.add_trace(go.Scatter( x=x_line, y=rar_if(x_line), mode='lines', name='RAR IF',
    line=dict(color='magenta', width=2) ), row=1, col=1)

# --- Delta-family IF traces (slider-controlled) ---
delta_values = np.linspace(0.5, 10.0, 20)
n_delta = len(delta_values)

# delta_traces = []
# for j, delta_val in enumerate(delta_values):
#     trace = go.Scatter(
#         x=x_line,
#         y=delta_family(delta_val, x_line),
#         mode='lines',
#         line=dict(color='red', width=2),
#         name=f"δ = {delta_val:.2f}",
#         visible=False,
#         legendgroup="delta_if",
#         showlegend=True
#     )
#     delta_traces.append(trace)
#     fig.add_trace(trace, row=1, col=1)

# Keep track of how many fixed traces we have before per-galaxy data
n_fixed_traces = 1 + 4 + n_delta  # background + y=x + 3 special IFs + 10 delta IFs (15 in total)

# Add one highlighted trace per galaxy
for gal in galaxies:
    vals = sparc_data[gal]
    # --- Subplot (1,1): Highlighted points on RAR ---
    fig.add_trace(go.Scatter(
        x=vals['g_bar'], y=vals['g_obs'],
        mode='markers', marker=dict(size=8, color='blue'),
        name=f"{gal} RAR",
        visible=(gal == galaxies[0]),
        legendgroup="rar",
        showlegend=False
    ), row=1, col=1)

    # --- Subplot (1,2): Rotation Curves ---
    # Rotation curve: Vobs
    fig.add_trace(go.Scatter(
        x=vals['r'], y=vals['Vobs'],
        mode='lines+markers', line=dict(color='black'),
        name=f"{gal} Vobs/g_obs",
        visible=(gal == galaxies[0]),
        legendgroup="rc",
        showlegend=True
    ), row=1, col=2)
    # Rotation curve: Vbar
    fig.add_trace(go.Scatter(
        x=vals['r'], y=vals['Vbar'],
        mode='lines+markers', line=dict(color='firebrick'),
        name=f"{gal} Vbar/g_bar",
        visible=(gal == galaxies[0]),
        legendgroup="rc",
        showlegend=True
    ), row=1, col=2)

    # --- Subplot (1,2): 'Acceleration Curves' ---
    # Acceleration curve: g_obs
    fig.add_trace(go.Scatter(
        x=vals['r'], y=vals['g_obs'],
        mode='lines+markers', line=dict(color='black'),
        name=f"{gal} g_obs",
        visible=(gal == galaxies[0]),
        legendgroup="acc",
        showlegend=False
    ), row=2, col=2)
    # Acceleration curve: g_bar
    fig.add_trace(go.Scatter(
        x=vals['r'], y=vals['g_bar'],
        mode='lines+markers', line=dict(color='firebrick'),
        name=f"{gal} g_bar",
        visible=(gal == galaxies[0]),
        legendgroup="acc",
        showlegend=False
    ), row=2, col=2)

# --- Build visibility grid: vis_grid[g_idx][d_idx] ---
vis_grid = []
for g_idx in range(galaxy_count):
    vis_list_per_delta = []
    for d_idx in range(n_delta):
        vis = [True] * (n_fixed_traces - n_delta)   # base IF lines, no delta_IF yet

        # # delta_IF visibility: only show the selected delta
        # vis += [(j == d_idx) for j in range(n_delta)]

        # galaxy-specific RAR traces
        for gg in range(galaxy_count):
            vis += [gg == g_idx] * 5    # 5 traces per galaxy (RAR, RC, Acc)

        vis_list_per_delta.append(vis)
    vis_grid.append(vis_list_per_delta)

buttons = []
for g_idx, gal in enumerate(galaxies):
    buttons.append(dict(
        label=gal,
        method='update',
        args=[{'visible': vis_grid[g_idx][3]}],  # default delta = 2.0
    ))

steps = []
# for d_idx, d in enumerate(delta_values):
#     steps.append(dict(
#         label=f"{d:.2f}",
#         method='update',
#         args=[{'visible': vis_grid[table["Galaxy"].tolist().index("UGC06787")][d_idx]}]     # default galaxy = UGC 6787
#     ))


exponents = np.arange(-13, -7)
tickvals = [10.0**e for e in exponents]
ticktext = [f"10{superscript_number(e)}" for e in exponents]

# --- Layout ---
fig.update_layout(
    updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                      x=1.05, y=1.1, xanchor='left', yanchor='top')],
    # sliders=[dict( active=3, currentvalue={"prefix": "δ = "}, pad={"t": 50}, steps=steps)],
    title=f"RAR & Rotation Curves of SPARC galaxies",
    height=700, width=900,
    xaxis=dict(title=f"g_bar (ms{superscript_number(-2)})", type='log',
               tickvals=tickvals, ticktext=ticktext),
    yaxis=dict(title=f"g_obs (ms{superscript_number(-2)})", type='log',
               tickvals=tickvals, ticktext=ticktext),
    xaxis2=dict(title=" "),  # RC X-axis
    yaxis2=dict(title=f"Velocities (kms{superscript_number(-1)})"),  # RC Y-axis
    xaxis3=dict(title="Radius (kpc)"),  # Acc X-axis
    yaxis3=dict(title=f"g (ms{superscript_number(-2)})", type='log',
               tickvals=tickvals, ticktext=ticktext),  # Acc Y-axis
    legend=dict(x=1.05, y=1, xanchor='left')
)

# fig.update_traces(visible=False)
# for i, v in enumerate(vis_grid[0][0]):
#     fig.data[i].visible = v

fig.write_html("/mnt/users/koe/plots/RAR.html")

# TO DO:
# 1. Add IFs to RC and Acc plots. Maybe add some sort of residuals too?
# 2. Add delta family and start looking at features.
