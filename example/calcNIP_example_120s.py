import sys
import datetime
import numpy as np
import obspy
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
from matplotlib.ticker import *
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



if __name__ == '__main__':

    polarization = 'retrograde'
    Fs = 20.0 ### sampling frequency [Hz]
    azimuth = 220.0 ### propagating direction of Rayleigh waves (crater->seismometer)
    windL = 120.0 ### time window length [sec]
    station = 'V.KIRA'
    #starttime = datetime.datetime(2017,1,1,1,10,0)
    starttime = datetime.datetime(2017,10,15,1,10,0)

    stream_Z = obspy.read('sac/'+starttime.strftime("%Y%m%d")+'/'+starttime.strftime("%Y%m%d%H")+'00'+station+'.U.sac').resample(Fs, window='hann')
    stream_N = obspy.read('sac/'+starttime.strftime("%Y%m%d")+'/'+starttime.strftime("%Y%m%d%H")+'00'+station+'.N.sac').resample(Fs, window='hann')
    stream_E = obspy.read('sac/'+starttime.strftime("%Y%m%d")+'/'+starttime.strftime("%Y%m%d%H")+'00'+station+'.E.sac').resample(Fs, window='hann')


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
    Shift vertical component and Calculate NIP(t,f)
    """
    Sv_shifted = filt_par.get_shift(polarization) * Sv
    nip = filt_par.NIP(Sr, Sv_shifted)
    nip_mean = np.nanmean(nip, axis=1) ### averege over time
    nip_eightyper = np.percentile(nip, 80, axis=1) ### 80% percentile over time

    """
    Plot
    """
    # 図のアスペクト比を柔軟に変更できるように設定
    # デフォルトは (width=6, height=8) だが、必要に応じて変更可能
    figure_width = 6    # 図の幅 [インチ]
    figure_height = 8   # 図の高さ [インチ]
    figsize = (figure_width, figure_height)
    
    x_lim = 0.12
    fig = plt.figure(figsize=figsize) 
    ax1 = plt.axes([x_lim,0.82,0.68,0.15])
    trace_amp = 1.5*np.max(np.abs(tr_Z))
    ax1.plot(np.linspace(0,windL, len(tr_Z)), tr_Z, lw=1, color='k')
    ax1.annotate('Z (raw)', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                horizontalalignment='left', verticalalignment='top')
    ax1.set_xlim(0,windL)
    ax1.set_ylim(-trace_amp, trace_amp)
    ax1.set_ylabel('', fontsize=14)

    ax11 = plt.axes([0.9,0.82,0.1,0.2])
    plt.xticks(())
    plt.yticks(())
    ax11.spines['right'].set_color('none')
    ax11.spines['left'].set_color('none')
    ax11.spines['top'].set_color('none')
    ax11.spines['bottom'].set_color('none')   
    ax11.patch.set_alpha(0.)



    ax5 = plt.axes([x_lim,0.55,0.85,0.2])
    SC = ax5.pcolormesh(Tv, Fv, np.abs(Sv), cmap=plt.cm.jet, rasterized=True)
    ax5.set_xlim(0,windL)
    ax5.set_ylim(0.05,10)
    ax5.set_yscale('log')
    ax5.set_ylabel('Frequency [Hz]', fontsize=14)

    ax6 = plt.axes([0.9,0.55,0.1,0.2])
    plt.xticks(())
    plt.yticks(())
    ax6.spines['right'].set_color('none')
    ax6.spines['left'].set_color('none')
    ax6.spines['top'].set_color('none')
    ax6.spines['bottom'].set_color('none')   
    ax6.patch.set_alpha(0.)
    cbar=plt.colorbar(SC, pad=0.05, orientation='vertical')
    cbar.set_label(r'$|S_v(\tau, f)|$ [m/s]', fontsize=14)
    cbar.ax.tick_params(labelsize=12)


    ax3 = plt.axes([x_lim,0.3,0.85,0.2])      
    import matplotlib as mpl
    cmap = plt.cm.bwr
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    bounds = np.arange(-1, 1.25, 0.25) 
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    SC = ax3.pcolormesh(Tv, Fv, nip, cmap=cmap, norm=norm, rasterized=True)


    ax3.set_xlim(0,windL)
    ax3.set_ylim(0.05,10)
    ax3.set_yscale('log')
    ax3.set_xlabel('lapse time [s]', fontsize=14)
    ax3.set_ylabel('Frequency [Hz]', fontsize=14)

    ax4 = plt.axes([0.9,0.3,0.1,0.2])
    #ax4.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    plt.xticks(())
    plt.yticks(())
    ax4.spines['right'].set_color('none')
    ax4.spines['left'].set_color('none')
    ax4.spines['top'].set_color('none')
    ax4.spines['bottom'].set_color('none')   
    ax4.patch.set_alpha(0.)
    cbar=plt.colorbar(SC, pad=0.05, orientation='vertical', ticks=np.arange(-1, 1.25, 0.25))
    cbar.set_label('NIP', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    
    ax7 = plt.axes([x_lim,0.07,0.3,0.15])   
    ax7.plot(nip_mean, Fv[:,0], lw=1.5, ls='--', color='C0', label='NIP (mean)', zorder=2)
    ax7.plot(nip_eightyper, Fv[:,0], lw=1.5, color='C1', label='NIP (80th percentile)', zorder=2)
    ax7.set_ylim(0.05,10)
    ax7.set_yscale('log')
    ax7.set_xlabel('NIP', fontsize=14)
    ax7.set_ylabel('Frequency [Hz]', fontsize=14)
    ax7.legend(fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, )
    ax7.set_xlim(-1,1)
    
    
    ax1 = apply_gmt_style_to_axis(ax1, grid=False, minor_ticks=True)
    ax3 = apply_gmt_style_to_axis(ax3, grid=False, minor_ticks=True)
    ax5 = apply_gmt_style_to_axis(ax5, grid=True, minor_ticks=True)
    ax7 = apply_gmt_style_to_axis(ax7, grid=True, minor_ticks=True)

    plt.savefig("example.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("example.png", dpi=300, bbox_inches='tight')
    plt.show()