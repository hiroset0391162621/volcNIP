import sys
import datetime
import numpy as np
import obspy
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
from matplotlib.ticker import *
import warnings
warnings.simplefilter('ignore')
sys.path.append(sys.path.append("core/"))
import backazimuth
sys.path.append(sys.path.append("utils/"))
import circular
import bootstrap    

try:
    import scienceplots
except:
    pass
plt.style.use(['science', 'nature'])
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.linewidth'] = 1.5 
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.minor.width'] = 1.5
plt.rcParams['xtick.direction'] = "inout"
plt.rcParams['ytick.direction'] = "inout"
plt.rcParams['xtick.major.size'] = 6.0
plt.rcParams['xtick.minor.size'] = 4.0
plt.rcParams['ytick.major.size'] = 6.0
plt.rcParams['ytick.minor.size'] = 4.0
plt.rcParams['xtick.major.pad']='8'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.edgecolor'] = '#08192D' 
plt.rcParams['axes.labelcolor'] = '#08192D' 
plt.rcParams['xtick.color'] = '#08192D' 
plt.rcParams['ytick.color'] = '#08192D'
plt.rcParams['text.color'] = '#08192D' 
plt.rcParams['legend.framealpha'] = 1.0 
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex']   = False



from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
import warnings
warnings.filterwarnings('ignore')


gscolors = np.array(
    [
        [243, 243, 241, 1],
        [0, 242, 242, 1],
        [0, 121, 242, 1],
        [0, 0, 242, 1],
        [121, 0, 242, 1],
        [242, 0, 242, 1],
        [242, 0, 121, 1],
        [242, 0, 0, 1],
        [242, 121, 0, 1],
        [242, 242, 0, 1],
        [121, 242, 0, 1],
        [0, 242, 0, 1],
        [0, 242, 121, 1],
        [0, 242, 242, 1],
        [243, 243, 241, 1],
    ],
    dtype=np.float64,
)
gscolors[:, :3] /= 256  
gscmap = ListedColormap(gscolors)
hsv2 = LinearSegmentedColormap.from_list("gscmap2", colors=gscolors)


def ci_eachf(Fv, baz, Tv):
    time_baz_idx = np.where( (Tv[0,:]>=0) & (Tv[0,:]<=120) )[0]
    mean_direction_ci = np.zeros((Fv.shape[0], 3))*np.nan
    for i in range(Fv.shape[0]): 
        baz2 = (baz[:,time_baz_idx][i,:]).flatten()
        baz2 = baz2[baz2==baz2]
        
        
        
        if Fv[i,0]>=0.0:
            mean_direction_low, mean_direction_high, kappa_low, kappa_high = bootstrap.bootstrap_vm_confidence_interval(np.deg2rad(baz2))   
            mean_direction_low = np.rad2deg(mean_direction_low)
            mean_direction_high = np.rad2deg(mean_direction_high)
            print(str(Fv[i,0])+"Hz", mean_direction_low, mean_direction_high)
            
            if mean_direction_low<-180:
                mean_direction_low = np.nan
            
            if mean_direction_high>180:
                mean_direction_high = np.nan
                
            mean_direction_ci[i,0] = Fv[i,0]
            mean_direction_ci[i,1] = mean_direction_low
            mean_direction_ci[i,2] = mean_direction_high
        

    return mean_direction_ci

def plot_baz(windL, tr_Z, Fv, Tv, Sv, baz, nip):
    fig = plt.figure(figsize=(5,8)) 


    ax1 = plt.axes([0.1,0.78,0.68,0.15])
    trace_amp = 1.5*np.max(np.abs(tr_Z))
    ax1.plot(np.linspace(0,windL, len(tr_Z)), tr_Z, lw=1, color='k')
    ax1.annotate(station+'.U (raw)', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                horizontalalignment='left', verticalalignment='top')
    ax1.set_xlim(0,windL)
    ax1.set_ylim(-trace_amp, trace_amp)
    ax1.set_ylabel('velocity [m/s]', fontsize=12)

    ax11 = plt.axes([0.9,0.5,0.1,0.2])
    plt.xticks(())
    plt.yticks(())
    ax11.spines['right'].set_color('none')
    ax11.spines['left'].set_color('none')
    ax11.spines['top'].set_color('none')
    ax11.spines['bottom'].set_color('none')   
    ax11.patch.set_alpha(0.)



    ax5 = plt.axes([0.1,0.5,0.85,0.2])
    SC = ax5.pcolormesh(Tv, Fv, np.abs(Sv), cmap=plt.cm.jet, rasterized=True)
    ax5.set_xlim(0,windL)
    ax5.set_ylim(0.05,10)
    ax5.set_yscale('log')
    ax5.set_ylabel('Frequency [Hz]', fontsize=12)

    ax6 = plt.axes([0.9,0.5,0.1,0.2])
    #ax4.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    plt.xticks(())
    plt.yticks(())
    ax6.spines['right'].set_color('none')
    ax6.spines['left'].set_color('none')
    ax6.spines['top'].set_color('none')
    ax6.spines['bottom'].set_color('none')   
    ax6.patch.set_alpha(0.)
    cbar=plt.colorbar(SC, pad=0.05, orientation='vertical')
    cbar.set_label(r'$|S_v(\tau, f)|$ [m/s]', fontsize=12)
    cbar.ax.tick_params(labelsize=12)


    ax2 = plt.axes([0.1,0.25,0.85,0.2])      
    bounds = [-30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    SC = ax2.pcolormesh(Tv, Fv, baz, cmap=hsv2, norm=norm, rasterized=True)


    freq = Fv[:,0]
    freq_low, freq_high = 1.3, 3
    i_freq_low = np.argmin(np.abs(freq - freq_low))
    i_freq_high = np.argmin(np.abs(freq - freq_high))
    print('frequency', Fv)
    print('df', np.diff(freq))
    print('time', Tv[0,:])
    print('dt', np.diff(Tv[0,:]))

    ax2.set_xlim(0,windL)
    ax2.set_ylim(0.05,10)
    ax2.set_yscale('log')
    #ax2.set_xlabel('lapse time [s]', fontsize=12)
    ax2.set_ylabel('Frequency [Hz]', fontsize=12)


    ax4 = plt.axes([0.9,0.25,0.1,0.2])
    #ax4.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    plt.xticks(())
    plt.yticks(())
    ax4.spines['right'].set_color('none')
    ax4.spines['left'].set_color('none')
    ax4.spines['top'].set_color('none')
    ax4.spines['bottom'].set_color('none')   
    ax4.patch.set_alpha(0.)
    cbar=plt.colorbar(SC, pad=0.05, orientation='vertical')
    cbar.set_label(r'Back Azimuth [$^{\circ}$]', fontsize=12)
    cbar.ax.tick_params(labelsize=12)



    ax3 = plt.axes([0.1,0.0,0.85,0.2])
    import matplotlib as mpl
    cmap = plt.cm.binary
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    bounds = np.arange(0.8, 1.02, 0.02) 
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    SC = ax3.pcolormesh(Tv, Fv, nip, cmap=cmap, norm=norm, rasterized=True)


    ax3.set_xlim(0,windL)
    ax3.set_ylim(0.05,10)
    ax3.set_yscale('log')
    ax3.set_xlabel('lapse time [s]', fontsize=12)
    ax3.set_ylabel('Frequency [Hz]', fontsize=12)

    ax4 = plt.axes([0.9,0.0,0.1,0.2])
    #ax4.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    plt.xticks(())
    plt.yticks(())
    ax4.spines['right'].set_color('none')
    ax4.spines['left'].set_color('none')
    ax4.spines['top'].set_color('none')
    ax4.spines['bottom'].set_color('none')   
    ax4.patch.set_alpha(0.)
    cbar=plt.colorbar(SC, pad=0.05, orientation='vertical', ticks=[0.4,0.5,0.6,0.7,0.8,0.9,1], extend='min')
    cbar.set_label('NIP', fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    plt.suptitle(starttime.strftime("%Y.%m.%d %H:%M:%S")+'-'+(starttime+datetime.timedelta(seconds=windL)).strftime(" %H:%M:%S"), fontsize=14)

    
    plt.show()
    
    
def plot_bazmean(windL, Tv, Fv, mean_direction_ci):
    time_baz_idx = np.where( (Tv[0,:]>=0) & (Tv[0,:]<=windL) )[0]
    mean_baz2 = np.zeros(Fv.shape[0])*np.nan
    Results = np.zeros((Fv.shape[0],2))
    for i in range(Fv.shape[0]):
        baz2 = (baz[:,time_baz_idx][i,:]).flatten()
        baz2 = baz2[baz2==baz2]
        circmean_val = circular.mean(baz2, deg=True)
        if circmean_val>180.0:
            circmean_val -= 360.0
        mean_baz2[i] = circmean_val
        Results[i] = np.array([Fv[i,0], circmean_val])

    fv = Results[:,0]
    x = Results[:,1]

    fig = plt.figure(figsize=(5,8)) 
    ax3 = plt.axes([0.1,0.68,0.4,0.25])
    plt.plot(x, fv, lw=1.5, color='C0', zorder=2)
    
    for i in range(len(mean_direction_ci[:,0])):
        plt.plot([mean_direction_ci[i,1], mean_direction_ci[i,2]], [mean_direction_ci[i,0], mean_direction_ci[i,0]], color='pink', zorder=1)
    
    plt.axvline(x=30.0, color='red', ls='--', zorder=3) ### V.KIRA -> Shinmoe 
    plt.axvline(x=50.0, color='red', ls='--', zorder=3) ### V.KIRA -> Shinmoe 
    plt.yscale('log')
    plt.xlim(-180,180)
    plt.xticks([-180,-90,0,90,180])
    plt.ylim(0.05,10)
    ax3.set_xlabel(r'mean Back Azimuth [$^{\circ}$]', fontsize=12)
    ax3.set_ylabel('Frequency [Hz]', fontsize=12)
    ax3.annotate('Shinmoe-dake', xy=(40,10), xycoords='data', fontsize=10, va='bottom', ha='center', xytext=(0, 3), textcoords='offset points', color='red')


    ax4 = plt.axes([0.65,0.73,0.2,0.2])
    ax4.plot(x, fv, lw=1.5, color='C0')
    for i in range(len(mean_direction_ci[:,0])):
        plt.plot([mean_direction_ci[i,1], mean_direction_ci[i,2]], [mean_direction_ci[i,0], mean_direction_ci[i,0]], color='pink', zorder=1)
    ax4.set_yscale('log')
    ax4.set_xlim(30,50)
    ax4.set_ylim(1,3)
    #ax4.set_xlabel(r'mean Back Azimuth [$^{\circ}$]', fontsize=10)


    plt.suptitle(starttime.strftime("%Y.%m.%d %H:%M:%S")+'-'+(starttime+datetime.timedelta(seconds=windL)).strftime(" %H:%M:%S"), fontsize=14)
        
    plt.show()

    


if __name__ == '__main__':

    Fs = 20.0 ### sampling frequency [Hz]
    azimuth = 40.0 ### initial value. does not affect the final result.
    windL = 120.0 ### time window length [sec]
    station = 'V.KIRA'
    starttime = datetime.datetime(2017,1,1,1,10,0)
    #starttime = datetime.datetime(2017,10,15,1,10,0)

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
    Back azimuth estimation
    """
    baz, nip, Tv, Fv, Sv, Se, Sn = backazimuth.calc_baz(tr_Z.copy(), tr_E.copy(), tr_N.copy(), Fs, azimuth, 'retrograde')

    plot_baz(windL, tr_Z, Fv, Tv, Sv, baz, nip)


    
    """
    bootstrap
    Note: Applying the bootstrap method to all f is very computationally expensive.
    """
    mean_direction_ci = ci_eachf(Fv, baz, Tv)
    
    plot_bazmean(windL, Tv, Fv, mean_direction_ci)
    
        
    

    