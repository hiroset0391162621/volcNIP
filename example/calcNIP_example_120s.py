
import sys
import datetime
import numpy as np
import obspy
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
from matplotlib.ticker import *
from matplotlib.gridspec import GridSpec
import warnings
warnings.simplefilter('ignore') 
sys.path.append(sys.path.append("nipfilter/"))
from core import stransform, istransform
import filter as filt_par

from typing import Optional, Tuple, Union, List, Any, Dict, Callable, TypeVar, cast
import warnings
warnings.filterwarnings('ignore')

try:
    import scienceplots  # type: ignore
except:
    pass

# GMT-style matplotlib settings
try:
    plt.style.use(["science", "nature"])
except:
    plt.style.use("default")  # Fallback to default style

# GMT-style color and font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# GMT-style tick and axis settings
plt.rcParams['xtick.direction'] = "out"
plt.rcParams['ytick.direction'] = "out"
plt.rcParams["text.usetex"] = False
plt.rcParams['agg.path.chunksize'] = 100000
plt.rcParams["date.converter"] = "concise"

# Tick parameters (GMT-like style)
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['axes.linewidth'] = 1.0  # GMT uses thinner lines
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['xtick.major.size'] = 4.0
plt.rcParams['xtick.minor.size'] = 2.0
plt.rcParams['ytick.major.size'] = 4.0
plt.rcParams['ytick.minor.size'] = 2.0

# GMT-style colors
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

# Grid settings (GMT-like)
plt.rcParams['axes.grid'] = False  # We'll add custom grid
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.7

def apply_gmt_style_to_axis(ax: Any, grid: bool = True, minor_ticks: bool = True) -> Any:
    """
    Apply GMT-style formatting to matplotlib axis
    
    Args:
        ax: Matplotlib axis
        grid: Whether to show grid
        minor_ticks: Whether to show minor ticks
        
    Returns:
        Formatted axis
    """
    # GMT-style frame
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')
    
    # GMT-style ticks
    ax.tick_params(axis='x', which='major', labelsize=13, width=1.0, length=4.0, direction='out', color='black')
    ax.tick_params(axis='y', which='major', labelsize=14, width=1.0, length=4.0, direction='out', color='black')
    if minor_ticks:
        ax.tick_params(axis='both', which='minor', width=0.5, length=2.0, direction='out', color='black')
    
    # GMT-style grid
    if grid:
        ax.grid(True, which='major', color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
    
    # White background
    ax.set_facecolor('white')
    
    return ax


def create_nip_plot(tr_Z, tr_N, tr_E, Tv, Fv, Sv, nip, nip_mean, nip_eightyper,
                windL, figsize=(6, 8), save_path="example",
                cbar_width_ratio: float = 0.5, main_width_ratio: float = 6.0,
                hspace: float | None = None, wspace: float = 0.15,
                constrained: bool = False):
    """
    Plot NIP analysis results (subplot layout, aspect-ratio flexible).

    Args:
        tr_Z, tr_N, tr_E: Waveform arrays (vertical, north, east) in SI units.
        Tv, Fv: Time and frequency coordinate arrays from Stockwell transform.
        Sv: Stockwell transform (vertical component) complex values.
        nip: 2-D NIP values (frequency x time).
        nip_mean: Mean NIP over time (frequency vector).
        nip_eightyper: 80th percentile of NIP over time (frequency vector).
        windL: Time window length (seconds).
        figsize: Tuple (width, height) in inches for the figure size.
        save_path: Output filename prefix (without extension).
        cbar_width_ratio: Width ratio for colorbar column (default 0.5).
        main_width_ratio: Width ratio for main plot column (default 6.0).

    Returns:
        fig: Matplotlib Figure instance.
    """
    import matplotlib as mpl
    
    # Create figure (optionally enable constrained layout)
    fig = plt.figure(figsize=figsize, constrained_layout=constrained)
    
    # Flexible GridSpec. Narrow colorbars & controlled margins.
    # Clip extreme width settings provided by user.
    main_width_ratio = max(main_width_ratio, 1.0)
    cbar_width_ratio = max(min(cbar_width_ratio, main_width_ratio), 0.1)

    # Adaptive vertical spacing if not explicitly provided
    if hspace is None:
        # Increase spacing automatically for shorter figures to avoid overlap
        ref_height = 12.0  # reference height (inches)
        base_hspace = 0.35
        scale = ref_height / max(figsize[1], 1e-6)
        hspace = min(0.6, max(0.28, base_hspace * scale))

    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[1, 1.5, 1.5, 1],
        width_ratios=[main_width_ratio, cbar_width_ratio],
        hspace=hspace,
        wspace=wspace,
        left=0.15, right=0.82
    )
    # 1. Raw vertical waveform (top row)
    ax1 = fig.add_subplot(gs[0, 0])
    trace_amp = 1.5*np.max(np.abs(tr_Z))
    ax1.plot(np.linspace(0, windL, len(tr_Z)), tr_Z, lw=1, color='k')
    ax1.annotate('Z (raw)', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                horizontalalignment='left', verticalalignment='top')
    ax1.set_xlim(0, windL)
    ax1.set_ylim(-trace_amp, trace_amp)
    ax1.set_ylabel('Amplitude', fontsize=14)
    ax1.set_xlabel('lapse time [s]', fontsize=14)
    
    # 2. Stockwell transform amplitude (middle row)
    ax2 = fig.add_subplot(gs[1, 0])
    SC1 = ax2.pcolormesh(Tv, Fv, np.abs(Sv), cmap=plt.cm.jet, rasterized=True)
    ax2.set_xlim(0, windL)
    ax2.set_ylim(0.05, 10)
    ax2.set_yscale('log')
    ax2.set_ylabel('Frequency [Hz]', fontsize=14)
    ax2.set_xlabel('lapse time [s]', fontsize=14)
    
    # 3. Colorbar for Stockwell amplitude
    ax2_cbar = fig.add_subplot(gs[1, 1])
    cbar1 = plt.colorbar(SC1, cax=ax2_cbar, orientation='vertical')
    # Vertical label
    cbar1.set_label(r'$|S_v(\tau, f)|$ [m/s]', fontsize=12, labelpad=15)
    cbar1.ax.tick_params(labelsize=10)
    
    # 4. NIP time-frequency image
    ax3 = fig.add_subplot(gs[2, 0])
    cmap = plt.cm.bwr
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    bounds = np.arange(-1, 1.25, 0.25) 
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    SC2 = ax3.pcolormesh(Tv, Fv, nip, cmap=cmap, norm=norm, rasterized=True)
    ax3.set_xlim(0, windL)
    ax3.set_ylim(0.05, 10)
    ax3.set_yscale('log')
    ax3.set_xlabel('lapse time [s]', fontsize=14)
    ax3.set_ylabel('Frequency [Hz]', fontsize=14)
    
    # 5. Colorbar for NIP
    ax3_cbar = fig.add_subplot(gs[2, 1])
    cbar2 = plt.colorbar(SC2, cax=ax3_cbar, orientation='vertical', 
                        ticks=np.arange(-1, 1.25, 0.25))
    # Vertical label
    cbar2.set_label('NIP', fontsize=12, labelpad=15)
    cbar2.ax.tick_params(labelsize=10)
    
    # 6. NIP summary statistics (mean & 80th percentile)
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(nip_mean, Fv[:,0], lw=1.5, ls='--', color='C0', label='mean', zorder=2)
    ax4.plot(nip_eightyper, Fv[:,0], lw=1.5, color='C1', label='80th percentile', zorder=2)
    ax4.set_ylim(0.05, 10)
    ax4.set_yscale('log')
    ax4.set_xlabel('NIP', fontsize=14)
    ax4.set_ylabel('Frequency [Hz]', fontsize=14)
    # Legend with shorter sample line (handlelength) and tighter spacing
    ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0,fontsize=12, handlelength=0.9, handletextpad=0.5, borderpad=0.3)
    ax4.set_xlim(-1, 1)
    
    # Apply GMT-like styling
    ax1 = apply_gmt_style_to_axis(ax1, grid=False, minor_ticks=True)
    ax2 = apply_gmt_style_to_axis(ax2, grid=True, minor_ticks=True)
    ax3 = apply_gmt_style_to_axis(ax3, grid=False, minor_ticks=True)
    ax4 = apply_gmt_style_to_axis(ax4, grid=True, minor_ticks=True)
    
    # Final layout adjustment (skip if constrained already took care of it)
    if not constrained:
        pad = 1.1 if figsize[1] < 9 else 0.9
        plt.tight_layout(pad=pad)
    
    # Save figure
    plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == '__main__':

    
    FIGURE_SIZE = (6, 8)  # change here; adaptive spacing should prevent overlap

    polarization = 'retrograde'
    Fs = 20.0  # sampling frequency [Hz]
    azimuth = 220.0  # propagation azimuth of Rayleigh waves (crater -> station)
    windL = 120.0  # time window length [s]
    
    stream_Z = obspy.read('sac/trZ.sac').resample(Fs, window='hann')
    stream_N = obspy.read('sac/trN.sac').resample(Fs, window='hann')
    stream_E = obspy.read('sac/trE.sac').resample(Fs, window='hann')

    starttime = stream_Z[0].stats.starttime

    stream_Z[0].data *= 1e-9
    stream_N[0].data *= 1e-9
    stream_E[0].data *= 1e-9


    stream_Z.trim(UTCDateTime(starttime), UTCDateTime(starttime)+windL)
    stream_N.trim(UTCDateTime(starttime), UTCDateTime(starttime)+windL)
    stream_E.trim(UTCDateTime(starttime), UTCDateTime(starttime)+windL)

    tr_Z = stream_Z[0].data
    tr_N = stream_N[0].data
    tr_E = stream_E[0].data
    
    """
    Stockwell transform
    """
    Sv, Tv, Fv = stransform(tr_Z, Fs, return_time_freq=True)
    Se, _, _ = stransform(tr_E, Fs, return_time_freq=True)
    Sn, _, _ = stransform(tr_N, Fs, return_time_freq=True)
    
    """
    Rotate horizontal components NE -> RT
    """
    Sr, St = filt_par.rotate_NE_RT(Sn, Se, azimuth)
    
    """
    Calculate NIP(t,f)
    """
    nip = filt_par.NIP(Sr, Sv, polarization=polarization, eps=None)
    nip_mean = np.nanmean(nip, axis=1)  # average over time
    nip_eightyper = np.percentile(nip, 80, axis=1)  # 80th percentile over time

    """
    Plot
    """

    fig = create_nip_plot(tr_Z, tr_N, tr_E, Tv, Fv, Sv, nip, nip_mean, nip_eightyper, windL, figsize=FIGURE_SIZE, save_path="example", cbar_width_ratio=0.3, main_width_ratio=7.0)
    
    
    plt.show()
    
    