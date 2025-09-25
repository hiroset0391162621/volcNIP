"""
NIP解析プログラム - 図のアスペクト比柔軟変更対応版（subplot + tight_layout版）

このプログラムでは、plt.subplotとplt.tight_layout()を使用することで：
1. 図のアスペクト比を柔軟に変更可能
2. サブプロットがはみ出ない自動レイアウト調整
3. どのようなアスペクト比でも適切に表示される

plt.axesの固定座標ではなく、GridSpecを使用した相対的なレイアウトにより、
図のサイズが変わってもレイアウトが自動調整されます。

アスペクト比の例:
- デフォルト（縦長）: figsize = (6, 8)
- 正方形: figsize = (8, 8) 
- 横長: figsize = (12, 6)
- より大きな図: figsize = (10, 12)
- コンパクト: figsize = (4, 6)
- 超横長: figsize = (16, 4)
"""

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
                   cbar_width_ratio: float = 0.5, main_width_ratio: float = 6.0):
    """
    NIP解析結果をプロットする関数 - subplot版（アスペクト比変更に対応）
    
    Args:
        tr_Z, tr_N, tr_E: 地震波形データ
        Tv, Fv: 時間・周波数軸
        Sv: 垂直成分のStockwell変換結果
        nip: NIP値
        nip_mean: NIP平均値
        nip_eightyper: NIP 80パーセンタイル値
        windL: 時間窓長
        figsize: 図のサイズ (width, height) のタプル
        save_path: 保存ファイル名のプレフィックス
        cbar_width_ratio: カラーバーの幅比率（デフォルトは0.5）
        main_width_ratio: メインプロットの幅比率（デフォルトは6.0）
    
    Returns:
        fig: matplotlib figure オブジェクト
    """
    import matplotlib as mpl
    
    # subplotを使用してレイアウトを自動調整
    fig = plt.figure(figsize=figsize)
    
    # GridSpecを使用してより柔軟なレイアウト
    # カラーバーを細くし、左右マージンを増やしてラベルがはみ出ないようにする
    # width比率はユーザー指定。極端な値はクリップ
    main_width_ratio = max(main_width_ratio, 1.0)
    cbar_width_ratio = max(min(cbar_width_ratio, main_width_ratio), 0.1)
    gs = fig.add_gridspec(4, 2,
                          height_ratios=[1, 1.5, 1.5, 1],
                          width_ratios=[main_width_ratio, cbar_width_ratio],
                          hspace=0.35, wspace=0.15,
                          left=0.15, right=0.82)  # 縦書きラベル用により右マージンを確保
    
    # 1. 波形プロット（上段全体）
    ax1 = fig.add_subplot(gs[0, 0])
    trace_amp = 1.5*np.max(np.abs(tr_Z))
    ax1.plot(np.linspace(0, windL, len(tr_Z)), tr_Z, lw=1, color='k')
    ax1.annotate('Z (raw)', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                horizontalalignment='left', verticalalignment='top')
    ax1.set_xlim(0, windL)
    ax1.set_ylim(-trace_amp, trace_amp)
    ax1.set_ylabel('Amplitude', fontsize=14)
    ax1.set_xlabel('lapse time [s]', fontsize=14)  # x軸ラベルを表示
    
    # 2. Stockwell変換結果プロット（中段左）
    ax2 = fig.add_subplot(gs[1, 0])
    SC1 = ax2.pcolormesh(Tv, Fv, np.abs(Sv), cmap=plt.cm.jet, rasterized=True)
    ax2.set_xlim(0, windL)
    ax2.set_ylim(0.05, 10)
    ax2.set_yscale('log')
    ax2.set_ylabel('Frequency [Hz]', fontsize=14)
    ax2.set_xlabel('lapse time [s]', fontsize=14)  # x軸ラベルを表示
    
    # 3. Stockwell変換のカラーバー（中段右）- 細くて見やすく
    ax2_cbar = fig.add_subplot(gs[1, 1])
    cbar1 = plt.colorbar(SC1, cax=ax2_cbar, orientation='vertical')
    # 従来通り横向き（縦書き）ラベル
    cbar1.set_label(r'$|S_v(\tau, f)|$ [m/s]', fontsize=12, labelpad=15)
    cbar1.ax.tick_params(labelsize=10)
    
    # 4. NIPプロット（中下段左）
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
    
    # 5. NIPカラーバー（中下段右）- 細くて見やすく
    ax3_cbar = fig.add_subplot(gs[2, 1])
    cbar2 = plt.colorbar(SC2, cax=ax3_cbar, orientation='vertical', 
                         ticks=np.arange(-1, 1.25, 0.25))
    # 従来通り横向き（縦書き）ラベル
    cbar2.set_label('NIP', fontsize=12, labelpad=15)
    cbar2.ax.tick_params(labelsize=10)
    
    # 6. NIP統計プロット（下段左）
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(nip_mean, Fv[:,0], lw=1.5, ls='--', color='C0', label='NIP (mean)', zorder=2)
    ax4.plot(nip_eightyper, Fv[:,0], lw=1.5, color='C1', label='NIP (80th percentile)', zorder=2)
    ax4.set_ylim(0.05, 10)
    ax4.set_yscale('log')
    ax4.set_xlabel('NIP', fontsize=14)
    ax4.set_ylabel('Frequency [Hz]', fontsize=14)
    ax4.legend(fontsize=12, loc='upper right')
    ax4.set_xlim(-1, 1)
    
    # GMT スタイル適用
    ax1 = apply_gmt_style_to_axis(ax1, grid=False, minor_ticks=True)
    ax2 = apply_gmt_style_to_axis(ax2, grid=True, minor_ticks=True)
    ax3 = apply_gmt_style_to_axis(ax3, grid=False, minor_ticks=True)
    ax4 = apply_gmt_style_to_axis(ax4, grid=True, minor_ticks=True)
    
    # レイアウトの自動調整 - ylabelがはみ出ないよう調整
    plt.tight_layout(pad=1.5)  # パディングを追加してラベルの余裕を確保
    
    # 図の保存
    plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == '__main__':

    
    FIGURE_SIZE = (5, 14)    

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
    Plot - 図のアスペクト比を柔軟に変更可能
    """
    # 上で設定したFIGURE_SIZEを使用して図を作成
    fig = create_nip_plot(tr_Z, tr_N, tr_E, Tv, Fv, Sv, nip, nip_mean, nip_eightyper, windL, figsize=FIGURE_SIZE, save_path="example", cbar_width_ratio=0.3, main_width_ratio=7.0)
    
    
    plt.show()
    
    