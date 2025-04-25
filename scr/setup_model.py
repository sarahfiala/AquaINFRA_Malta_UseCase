#%% Import libraries
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import flopy
import flopy.utils.binaryfile as bf
import xarray as xr
import pyvista as pv
import PVGeo
import yaml
#import imod

#%% Define directories and model name

# Base directory (inside Docker or Linux environment this would be "/app")
base_dir = os.path.abspath(os.path.dirname(__file__))  # This will be 'src/', so we go one level up
project_root = os.path.abspath(os.path.join(base_dir, ".."))

# Input, output, and model directories (relative to root)
input_dir = os.path.join(project_root, 'example_inputs')
output_dir = os.path.join(project_root, 'example_results')
model_name = 'Malta_Model'
model_dir = os.path.join(project_root, 'model_files', 'malta_simulation', model_name)

# Executables
imod_path = os.path.join(project_root, 'model_files', 'bin', '_imodwq', '_imodwq', 'imod-wq_svn392_x64r.exe')
seawat_exe_dir = os.path.join(project_root, 'model_files', 'bin', 'SEAWAT', 'SEAWAT', 'swt_v4_00_05')


# Create directories if they don't exist
for path in [input_dir, output_dir, model_dir]:
    os.makedirs(path, exist_ok=True)

#%% Load and process geological data from Petrel-exported CSV

# Define column names expected in the input file
fields = ['I', 'J', 'K', 'X', 'Y', 'Z', 'fac', 'por', 'hyd']

# Read CSV starting from the 11th line (skipping headers)
df_in = pd.read_csv(
    os.path.join(input_dir, r'C:\Users\ariel\Desktop\AquaINFRA_Malta_UseCase\example_inputs\petrel_data'),
    skiprows=11,
    delim_whitespace=True,
    names=fields,
    header=None,  
    index_col=False
)

# Round values to appropriate precision
df_in[['I', 'J', 'K']] = df_in[['I', 'J', 'K']].round(0)
df_in[['Z', 'fac', 'por', 'hyd']] = df_in[['Z', 'fac', 'por', 'hyd']].round(1)

#%% Generate model grid dimensions from the data

dx = dy = 200  # Horizontal cell size (in meters)
dz = 20        # Vertical layer thickness

# Extract unique layer, row, and column indices
lay_arr = np.unique(df_in.K.values)
row_arr = np.unique(df_in.I.values)
col_arr = np.unique(df_in.J.values)

# Initialize IBOUND array (used to define active/inactive cells)
full_ibound_arr = np.zeros((lay_arr.shape[0], row_arr.shape[0], col_arr.shape[0]))

# Extract top and bottom elevations from unique Z values
z_vals = np.unique(df_in.Z.values)
top_elev = np.nanmax(z_vals)
bot_elev = np.arange(top_elev, np.nanmin(z_vals) - 2 * dz, -dz)[1:]

# Print model grid info
print(f"Top elevation: {top_elev} m bsl")
print(f"Bottom elevation: {bot_elev[-1]} m bsl")
print(f"Number of layers: {full_ibound_arr.shape[0]}")
print(f"Rows: {full_ibound_arr.shape[1]}, Columns: {full_ibound_arr.shape[2]}")

#%% Define sea level changes and stress periods

# Sea levels (one for each stress period)
sp_sealevel = [
    -109.01, -109.01, -109.01, -109.01, -109.01, -109.01, -109.01, -109.01,
    -104.7775, -100.545, -96.3125, -92.08, -86.7325, -81.385, -76.0375,
    -70.69, -65.27, -59.85, -54.43, -49.01, -44.7025, -40.395, -36.0875,
    -31.78, -28.885, -25.99, -23.095, -20.2, -17.75, -15.3, -12.85,
    -10.4, -9.1375, -7.875, -6.6125, -5.35, -4.0125, -2.675, -1.3375, 0.0
]

# Create names for each stress period
sp_names = [f"malta_sp{i}" for i in range(len(sp_sealevel))]

# Duration of each stress period (in years)
sp_time = 500 * np.ones_like(sp_sealevel)

# Select the first stress period
a = 0
model_name_sp = sp_names[a]
sp_dir = os.path.join(model_dir, model_name_sp)
os.makedirs(sp_dir, exist_ok=True)

# Define time bounds for this mini-model run
time_start, time_end = 0, 19999
mini_modelname = f'SP_{time_start}_to_{time_end}'
mini_sp_dir = os.path.join(sp_dir, mini_modelname)

# Create SEAWAT model object
mswt = flopy.seawat.Seawat(mini_modelname, 'nam_swt', model_ws=mini_sp_dir, exe_name=seawat_exe_dir)

#%% Fill in hydraulic conductivity, porosity, and IBOUND arrays

# Initialize property arrays with shape of the model grid
ibound_arr = np.copy(full_ibound_arr)
hk_arr = np.copy(full_ibound_arr)
poro_arr = np.copy(full_ibound_arr)

# Filter only the cells with positive facies value
df_sel = df_in[df_in['fac'] > 0]

# Loop through each row and fill in the model arrays
for _, row in df_sel.iterrows():
    k, i, j = int(row['K']) - 1, int(row['I']) - 1, int(row['J']) - 1
    hk_arr[k, i, j] = row['hyd']
    poro_arr[k, i, j] = row['por']
    ibound_arr[k, i, j] = 1  # Active cell

#%% Define starting heads (strt_arr) and concentrations (sconc_arr)
sconc_arr = ibound_arr * 35.  # Starting concentration (35 TDS mg/l)
strt_arr = ibound_arr * 0    # Starting head (0 m)

#%% Create top and bottom elevation arrays for each cell

# # 1D arrays of top and bottom elevations for each layer
# top_arr_1d = np.arange(top_elev, bot_elev[-1], -dz)
# bot_arr_1d = np.arange(bot_elev[0], bot_elev[-1] - dz, -dz)

# # Initialize 3D arrays for full model grid
# arr_2d = np.ones((hk_arr.shape[1], hk_arr.shape[2]))
# top_arr_3d = np.ones_like(hk_arr)
# bot_arr_3d = np.ones_like(hk_arr)

# # Fill each layer with correct elevation values
# for lay in range(top_arr_1d.shape[0]):
#     top_arr_3d[lay, :, :] = arr_2d * top_arr_1d[lay]
#     bot_arr_3d[lay, :, :] = arr_2d * bot_arr_1d[lay]

# print("Top and bottom arrays created with shape:", top_arr_3d.shape)


# # Mask out the arrays where Hk = 0 m/d (inactive cells), replacing them with NaN
# ibound_arr_plot = np.copy(ibound_arr)
# ibound_arr_plot[ibound_arr == 0] = np.nan

# # Create an xarray from the ibound array - follows IMOD standards for 3D grid
# da = xr.DataArray(
#     data=ibound_arr_plot,  # ibound data with NaN for inactive cells
#     dims=["layer", "y", "x"],  # Define dimensions as layer, y, and x
#     coords={  # Define the coordinates for the grid
#         "top": (["layer", "y", "x"], top_arr_3d),  # Top elevation for each cell
#         "bottom": (["layer", "y", "x"], bot_arr_3d),  # Bottom elevation for each cell
#         "layer": np.arange(1, ibound_arr_plot.shape[0] + 1),  # Layer index (1-based)
#         "y": np.arange(1, 1 + ibound_arr_plot.shape[1]) * dy,  # y-coordinate values (scaled by dy)
#         "x": np.arange(1, 1 + ibound_arr_plot.shape[2]) * dx,  # x-coordinate values (scaled by dx)
#     }
# )

# # Create the voxels and plot the 3D grid using IMOD visualization
# z_grid = imod.visualize.grid_3d(
#     da, vertical_exaggeration=10, exterior_only=False, exterior_depth=1, return_index=False
# )

# # Start a new PyVista plotter to visualize the 3D grid
# plotter = pv.Plotter()

# # Add the 3D model from `z_grid` to the plotter (assumed to be a PyVista Grid)
# plotter.add_mesh(z_grid, show_edges=True)

# # Define coordinates for a plane at a specific z value (sea level)
# z_plane = -100  # Sea level in model units, with dz of 20m
# x_min, x_max = da.coords["x"].values.min(), da.coords["x"].values.max()
# y_min, y_max = da.coords["y"].values.min(), da.coords["y"].values.max()
# x_plane, y_plane = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
# z_plane_array = np.full_like(x_plane, z_plane)

# # Create a surface mesh for the defined plane using PyVista
# plane = pv.StructuredGrid(x_plane, y_plane, z_plane_array)

# # Add the plane to the plot with 50% transparency
# plotter.add_mesh(plane, color="blue", opacity=0.5)

# # Display the final plot
# plotter.show()

#%%  DIS Package - Fixed Order

# Define number of layers, rows, columns, and grid cell sizes
nlay = lay_arr.shape[0]
nrow = row_arr.shape[0]
ncol = col_arr.shape[0]
delr = dx  # Horizontal cell size
delc = dy  # Horizontal cell size

# Define the number of stress periods (should be defined based on sp_sealevel length)
nper = len(sp_sealevel)  # Number of stress periods

# Define stress period parameters
perlen = np.full(nper, 365.25 * 500)  # Array filled with 500 years in days for each stress period
nstp = np.full(nper, 50)  # Number of time steps per stress period

# Create the DIS package using Flopy
dis = flopy.modflow.ModflowDis(
    mswt, 
    nlay, 
    nrow, 
    ncol, 
    nper=nper, 
    delr=delr, 
    delc=delc, 
    top=top_elev, 
    botm=bot_elev, 
    perlen=perlen, 
    nstp=nstp)

#%% BAS Package 
bas = flopy.modflow.ModflowBas(mswt, ibound = ibound_arr, strt = strt_arr)

#%% LPF Package

# setting anisotropy to the hyd. conductivity field
vk_arr = hk_arr * 0.1 
lpf = flopy.modflow.ModflowLpf(mswt, laytyp = 0, hk = hk_arr, vka = vk_arr, ipakcb = 1)

#%% BOUNDARY CONDITIONS

# Copy ibound array for MT3DMS boundary usage
icbund_arr = ibound_arr.copy()

# Initialize input containers
ghb_input_all = []
ssmdata_all = []

# Get SSM itype dictionary
itype = flopy.mt3d.Mt3dSsm.itype_dict()

# Loop over each stress period
for a, sea_level in enumerate(sp_sealevel):

    ghb_input_lst = []
    ssmdata = []

    # Inland boundaries (edges of the domain, above sea level)
    row_edges = [0, nrow - 1]
    for i in range(ncol):
        for row in row_edges:
            col_cells = ibound_arr[:, row, i]
            if 1 in col_cells:
                lay_idx = col_cells.tolist().index(1)
                if bot_elev[lay_idx] + dz >= sea_level:
                    for k in range(nlay):
                        if bot_elev[k] + dz >= sea_level:
                            cond_val = hk_arr[k, row, i] * dz * 1000
                            ghb_input_lst.append([k, row, i, bot_elev[k] + dz, cond_val])
                            ssmdata.append([k, row, i, 0.0, itype['GHB']])
                            icbund_arr[k, row, i] = -1  # mark as GHB cell for MT3DMS

    # Offshore domain (top layer only, below sea level)
    for i in range(nrow):
        for j in range(ncol):
            col_cells = ibound_arr[:, i, j]
            if 1 in col_cells:
                lay_idx = col_cells.tolist().index(1)
                if bot_elev[lay_idx] + dz < sea_level:
                    cond_val = (vk_arr[lay_idx, i, j] * delc * delr) / dz
                    ghb_input_lst.append([lay_idx, i, j, sea_level, cond_val])
                    ssmdata.append([lay_idx, i, j, 35.0, itype['GHB']])

    # Store stress period data
    ghb_input_all.append(ghb_input_lst)
    ssmdata_all.append(ssmdata)

# Format GHB input for stress periods
ghb_arr_in = {
    d: [list(entry) for entry in ghb_input_all[d]] if ghb_input_all[d] else []
    for d in range(len(ghb_input_all))
}

# Format SSM input
ssm_arr_in = {
    e: ssmdata_all[e]
    for e in range(len(ssmdata_all))
}

# Create GHB Package
ghb = flopy.modflow.ModflowGhb(
    mswt,
    ipakcb=1,
    stress_period_data=ghb_arr_in
)

#%% RCH & DRN Package 

rch_val = 0.00125

# Create a 2D recharge array (for a single stress period)
rch_arr = np.zeros((ibound_arr.shape[1], ibound_arr.shape[2]))
drn_input_lst = []

# Apply recharge and define drains only for the top active cell in each column
for i in range(ibound_arr.shape[1]):
    for j in range(ibound_arr.shape[2]):
        lay_lst = [t for t in ibound_arr[:, i, j].tolist() if t == 1]
        if len(lay_lst) > 0:
            top_act_lay = ibound_arr[:, i, j].tolist().index(1)
            top_act_elev = top_elev - top_act_lay * dz

            # If top active elevation is above sea level
            if top_act_elev >= sea_level:
                # Assign recharge
                rch_arr[i, j] = rch_val

                # Assign drain
                cond_cell = (vk_arr[top_act_lay, i, j] * dx * dy) / dz
                drn_input_lst.append([int(top_act_lay), i, j, top_act_elev, cond_cell])

# Build recharge dictionary for each stress period
rch_arr_in = {}
for c in range(len(perlen)):
    rch_arr_in[c] = rch_arr

# Define RCH package
rch = flopy.modflow.ModflowRch(mswt, nrchop=3, ipakcb=1, rech=rch_arr_in)

# Define DRN package if applicable
if len(drn_input_lst) > 0:
    drn_arr_in = {}
    for c in range(len(perlen)):
        drn_arr_in[c] = drn_input_lst
    drn = flopy.modflow.ModflowDrn(mswt, ipakcb=1, stress_period_data=drn_arr_in)

#%% Output Control 
ihedfm = 1
iddnfm = 0
extension = ['oc', 'hds', 'ddn', 'cbc']
unitnumber = [14, 30, 52, 51]

# Output control settings
spd = {(0, 0): ['SAVE HEAD', 'SAVE BUDGET', 'PRINT HEAD', 'PRINT BUDGET',
                'SAVE HEADTEC', 'SAVE CONCTEC', 'SAVE VXTEC', 'SAVE VYTEC', 'SAVE VZTEC']}

for t in range(nper):
    spd[(t, nstp[t] - 1)] = ['SAVE HEAD', 'SAVE BUDGET', 'PRINT HEAD', 'PRINT BUDGET',
                             'SAVE HEADTEC', 'SAVE CONCTEC', 'SAVE VXTEC', 'SAVE VYTEC', 'SAVE VZTEC']

oc = flopy.modflow.ModflowOc(mswt, ihedfm=ihedfm, stress_period_data=spd,
                             unitnumber=unitnumber, compact=True)

#%% BTN Package
#   the BTN package
porosity = poro_arr
dt0 = 1000
nprs = 1
ifmtcn = 0
chkmas = False
nprmas = 10
nprobs = 10
total_sim_time=sum(perlen)
#timprs_lst = list(np.linspace(1,total_sim_time,(nper-1),endpoint=True,dtype=int))
timprs_lst = np.cumsum(perlen).tolist()
btn = flopy.mt3d.Mt3dBtn(mswt, nprs=nprs, timprs=timprs_lst, prsity=porosity, sconc=sconc_arr,
                         ifmtcn=ifmtcn, chkmas=chkmas, nprobs=nprobs, nprmas=nprmas,dt0=1000)
#%% ADV Package
adv = flopy.mt3d.Mt3dAdv(mswt, mixelm=0, mxpart=2000000)

#%% DSP Package
dmcoef = 0.0000864  # effective molecular diffusion coefficient [M2/D]
al = 1.  # longitudinal dispersivity (L)
trpt = 10  # transverse dispersivity for the cell in the x-direction (L)
trpv = 10  # transverse dispersivity for the cell in the z-direction (L)
dsp = flopy.mt3d.Mt3dDsp(mswt, al=al, trpt=trpt, trpv=trpv, dmcoef=dmcoef)

#%% VDF Package
iwtable = 0  # Flag for the table of densities (if using a density table or not)
densemin = 1000.  # Minimum density (typically freshwater in kg/m^3)
densemax = 1025.  # Maximum density (typically saline water in kg/m^3)
denseref = 1000.  # Reference density for freshwater
denseslp = 0.7143  # Slope of the density function for solute concentration (typically for seawater)
firstdt = 0.001  # First timestep in days
vdf = flopy.seawat.SeawatVdf(mswt, iwtable=iwtable, densemin=densemin, densemax=densemax,
                          denseref=denseref, denseslp=denseslp, firstdt=firstdt)

#%% SSM Package
#   write the SSM package
ssm_arrs={}
for c in range(nper):
    ssm_arrs[c]=np.copy(rch_arr_in[c])*0.0  # Recharge array, all set to 0
    
#ssm_rch_in_2d = ssm_rch_in.reshape(nrow, ncol)  # Reshape to 2D (255 x 135)

# Now broadcast to match the shape for all stress periods (nper, nrow * ncol)
#ssm_rch_all = np.broadcast_to(ssm_rch_in_2d, (nper, nrow, ncol))  # Shape (nper, 255, 135)

# Define the SSM package with the reshaped array for recharge
ssm = flopy.mt3d.Mt3dSsm(mswt, crch=ssm_arrs[0], stress_period_data=ssm_arr_in)

#%% Write simulation
mswt.write_input()


#%% Writing files
# Write the ascii file with vertical sum of active cells in IBOUND
full_ibound_arr = np.sum(ibound_arr, axis=0, dtype=np.int32)  # This is your 2D array (255, 135)

# Convert the 2D array to a string (join all rows together into one string)
full_ibound_arr_str = '\n'.join(' '.join(map(str, row)) for row in full_ibound_arr)

# Open the file in binary mode ('wb') to write the string as bytes (same as original behavior)
with open(os.path.join(mini_sp_dir, 'LOAD.ASC'), 'wb') as f:
    f.write(full_ibound_arr_str.encode('utf-8'))  # Encode the string into bytes before writing

# Create the pksf and pkst files - change it in case the grid discretization changes
pksf_lines = [
    'ISOLVER 1', 'NPC 2', 'MXITER 1500', 'RELAX .98', 'HCLOSEPKS 0.05', 'RCLOSEPKS 20000.0', 'PARTOPT 0',
    'PARTDATA', 'external 40 1. (free) -1', 
    'GNCOL {}'.format(ncol), 'GNROW {}'.format(nrow), 
    'GDELR', '{}'.format(dx), 'GDELC', '{}'.format(dy),  # Correct GDELC to use 'dy' for the vertical dimension
    'NOVLAPADV 2', 'END'
]

pkst_lines = [
    'ISOLVER 2', 'NPC 2', 'MXITER 1500', 'INNERIT 200', 'RELAX .98', 'RCLOSEPKS 1.0E-01',
    'HCLOSEPKS 1.0E+12', 'RELATIVE-L2NORM', 'END'
]

# Write pksf file
with open(os.path.join(mini_sp_dir, mini_modelname + '.pksf'), 'w') as f:
    for line in pksf_lines:
        f.write(line)
        f.write('\n')

# Write pkst file
with open(os.path.join(mini_sp_dir, mini_modelname + '.pkst'), 'w') as f:
    for line in pkst_lines:
        f.write(line)
        f.write('\n')

# Open the nam_swt file and append these three lines
nam_lines = [
    'PKSF 27 ' + mini_modelname + '.pksf', 
    'PKST 35 ' + mini_modelname + '.pkst', 
    'DATA 40 LOAD.ASC'
]

with open(os.path.join(mini_sp_dir, mini_modelname + '.nam_swt'), 'a') as f:
    for line in nam_lines:
        f.write(line)
        f.write('\n')

# Save the ibound_arr (if it doesn't exist)
ibound_arr_dir = os.path.join(mini_sp_dir, 'ibound_arr.npy')
if not os.path.isfile(ibound_arr_dir):
    np.save(ibound_arr_dir, ibound_arr)


#%% Writing the windows batch script
bat_path = r'C:\Users\ariel\Desktop\AquaINFRA_Malta_UseCase\runmod_parallel.bat'
with open(bat_path, 'w') as infile:
    infile.write("\"{}\" \"{}\\{}.nam_swt\"".format(imod_path, mini_sp_dir, mini_modelname))

print(f'Run file created at: {bat_path}')


