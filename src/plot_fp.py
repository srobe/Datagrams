import os
import math
import numpy as np
import datetime as dt
# matplotlib - MPLCONFIGDIR issue
try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
except:
    import tempfile
    import atexit
    import shutil

    mpldir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, mpldir) #rm dir on exit

    os.environ['MPLCONFIGDIR'] = mpldir

    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
# matplotlib display backend
#mpl.use('Agg')
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
#import matplotlib.pyplot as plt
# netCDF4 - PYTHON_EGG_CACHE issue
try:
    from netCDF4 import Dataset
except:
    import tempfile
    import atexit
    import shutil

    eggdir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, eggdir) # rm dir on exit

    os.environ['PYTHON_EGG_CACHE'] = eggdir

    from netCDF4 import Dataset
import multiprocessing as mp

import find

# custom cmap for all
from matplotlib.colors import LinearSegmentedColormap as lsc
m = lsc.from_list('name', ['white', 'saddlebrown'])

#==============================================================================

cmap = {
    'meteo':plt.cm.Greens,
    'oc':plt.cm.bone_r,
    'bc':plt.cm.bone_r,
    'ss':plt.cm.Purples,
    'du':plt.cm.Reds,
    'su':plt.cm.pink_r,
    'co':plt.cm.YlGn,
    'co2':plt.cm.YlGn,
    'nimass':plt.cm.Greens,
#    'all':plt.cm.copper_r,
    'all':m,
    'total':m,
    'dumass':plt.cm.Reds,
    'totmass':plt.cm.Reds,
    'ssmass':plt.cm.Purples,
    'bcmass':plt.cm.bone_r,
    'sumass':plt.cm.pink_r,
    'ocmass':plt.cm.bone_r,
}

width = 0.119
linewidth = 0.
global forecast

#==============================================================================

def rc():
    '''custom plot settings, use mpl.rcdefaults() to default'''
    mpl.rc('lines', linewidth=0.5, antialiased=True)
    mpl.rc('patch', linewidth=0.5, facecolor='348ABD', edgecolor='eeeeee', \
          antialiased=True)
    mpl.rc('axes', facecolor='w', edgecolor='black', linewidth=0.5)
    mpl.rc('font', family='sans-serif', size=10.0)
    mpl.rc('xtick', color='black')
    mpl.rc('xtick.major', size=4, pad=6)
    mpl.rc('xtick.minor', size=2, pad=6)
    mpl.rc('ytick', color='black')
    mpl.rc('ytick.major', size=4, pad=6)
    mpl.rc('ytick.minor', size=2, pad=6)
    mpl.rc('legend', fancybox=True, fontsize=10.0)
    mpl.rc('figure', figsize='8, 7.5', dpi=100, facecolor='white')
    #mpl.rc('figure', figsize='9.6, 9', dpi=100, facecolor='white') #88
    mpl.rc('figure.subplot', hspace=0.5, left=0.07, right=0.95, bottom=0.1, \
          top=0.95)

def custom_date_formatter(x, p):
    '''xaxis formatter for HHz + date + Year'''
    global forecast
    ti = mdates.num2date(x)
    if ti.hour == 0:
        if x < mdates.date2num(forecast + dt.timedelta(days=1)):
            return ti.strftime('%Hz\n%a %-d %b\n%Y')
        return ti.strftime('%Hz\n%a %-d %b')
    else:
        return ti.strftime('%Hz')

def do_plots(ipath, stations, fcst, opath):
    '''receive dict of stations'''
    if not os.path.exists(ipath):
        return
    s = []
    for sta in stations:
        s.append((
            ipath, sta, float(stations[sta]['lat']),
            float(stations[sta]['lon']), fcst, opath
        ))

    #import time
    #start = time.time()
    # parallelize across stations
    pool = mp.Pool(processes=28)
    pool.map(plot_wrapper, s)
    pool.close()
    pool.join()
    #print 'Elapsed Time: %.2fs' % ((time.time()-start))
    return

def plot_wrapper(args):
    return plot(*args)

def plot(ipath, station, lat, lon, fcst, opath):
    '''main plot driver'''
    global forecast
    forecast = fcst
    # plot settings
    rc()

    # read in all data for one station
    cldhgh = get_data(ipath, 'CLDHGH', station)
    cldmid = get_data(ipath, 'CLDMID', station)
    cldlow = get_data(ipath, 'CLDLOW', station)

    # 3d ext: could move to plotting routine since these depend upon product
    rh = get_data(ipath, 'RH', station)
    ocext = get_data(ipath, 'OCEXT', station)
    bcext = get_data(ipath, 'BCEXT', station)
    suext = get_data(ipath, 'SUEXT', station)
    co = get_data(ipath, 'CO', station)
    co2 = get_data(ipath, 'CO2', station)
    airdens = get_data(ipath, 'AIRDENS', station)
    ssext = get_data(ipath, 'SSEXT', station)
    duext = get_data(ipath, 'DUEXT', station)

    # 3d mass
    dumass = get_data(ipath, 'DU', station)
    ssmass = get_data(ipath, 'SS', station)
    ocmass = get_data(ipath, 'OC', station)
    bcmass = get_data(ipath, 'BC', station)
    sumass = get_data(ipath, 'SO4', station)
    nimass = get_data(ipath, 'NI', station)

    # 2d mass
    dus = get_data(ipath, 'DUSMASS25', station)
    sss = get_data(ipath, 'SSSMASS25', station)
    ocs = get_data(ipath, 'OCSMASS', station)
    bcs = get_data(ipath, 'BCSMASS', station)
    nis = get_data(ipath, 'NISMASS25', station)
    so4 = get_data(ipath, 'SO4SMASS', station)

    u = get_data(ipath, 'U', station)
    v = get_data(ipath, 'V', station)

    slp = get_data(ipath, 'SLP', station)
    t2m = get_data(ipath, 'T2M', station)

    ssexttau = get_data(ipath, 'SSEXTTAU', station)
    duexttau = get_data(ipath, 'DUEXTTAU', station)
    bcexttau = get_data(ipath, 'BCEXTTAU', station)
    ocexttau = get_data(ipath, 'OCEXTTAU', station)
    suexttau = get_data(ipath, 'SUEXTTAU', station)
    niexttau = get_data(ipath, 'NIEXTTAU', station)

    prectot = get_data(ipath, 'PRECTOT', station)
    precsno = get_data(ipath, 'PRECSNO', station)
    preccon = get_data(ipath, 'PRECCON', station)
    u2m = get_data(ipath, 'U2M', station)
    v2m = get_data(ipath, 'V2M', station)

    cosc = get_data(ipath, 'COSC', station)

    ###########################################################################

    # pressure level axis
    yi = [i for i in range(len(rh['lev'])) if 600. <= rh['lev'][i] <= 950.]

    # time axes
    cldhgh_t = [fcst+dt.timedelta(seconds=int(item)) for item in cldhgh['time']]
    precip_t = [fcst+dt.timedelta(seconds=int(item)) for item in prectot['time']]
    rh_t = [fcst+dt.timedelta(seconds=int(t)) for t in rh['time']]
    x = np.arange(len(rh_t))
    x = [fcst+dt.timedelta(hours=int(ti) * 3) for ti in x] # 3-hourly data!
    xi = mdates.date2num(x)
    x, y = np.meshgrid(x, yi)

    ###########################################################################

    # precip - correct units (goal: mm?)

    # scaling for imagery
    cldhgh['data'] = np.absolute(cldhgh['data'])             # %
    cldmid['data'] = np.absolute(cldmid['data'])             # %
    cldlow['data'] = np.absolute(cldlow['data'])             # %
    precsno['data'] = precsno['data']*1e4                    # kg m-2 s-1
    prectot['data'] = prectot['data']*1e4 - precsno['data']  # kg m-2 s-1
    preccon['data'] = preccon['data']*1e4                    # kg m-2 s-1

    rh['data'] = rh['data'].transpose()[yi]*100              # %
    ocext['data'] = ocext['data'].transpose()[yi]*1e3        # km-1
    bcext['data'] = bcext['data'].transpose()[yi]*1e3        # km-1
    ssext['data'] = ssext['data'].transpose()[yi]*1e3        # km-1
    duext['data'] = duext['data'].transpose()[yi]*1e3        # km-1
    suext['data'] = suext['data'].transpose()[yi]*1e3        # km-1
    co['data'] = co['data'].transpose()[yi]                  # mol mol-1
    co2['data'] = co2['data'].transpose()[yi]                # mol mol-1
    airdens['data'] = airdens['data'].transpose()[yi]*1e9    # ug m-3

    u['data'] = u['data'].transpose()[yi]*1.94384            # knots
    v['data'] = v['data'].transpose()[yi]*1.94384            # knots

    dumass['data'] = dumass['data'].transpose()[yi]          # kg kg-1
    ssmass['data'] = ssmass['data'].transpose()[yi]          # kg kg-1
    ocmass['data'] = ocmass['data'].transpose()[yi]          # kg kg-1
    bcmass['data'] = bcmass['data'].transpose()[yi]          # kg kg-1
    sumass['data'] = sumass['data'].transpose()[yi]          # kg kg-1
    nimass['data'] = nimass['data'].transpose()[yi]          # kg kg-1

    no3_coeff = 80.043 / 62.0
    so4_coeff = 132.14 / 96.06

    totmass = dict(dumass)
    totmass['field'] = 'TOT'
    totmass['data'] = dumass['data'] \
                   + ssmass['data'] \
                   + ocmass['data'] \
                   + bcmass['data'] \
                   + sumass['data'] * so4_coeff \
                   + nimass['data'] * no3_coeff

    u2m['data'] = (u2m['data']**2)*(1.94384**2)              # knots^2
    v2m['data'] = (v2m['data']**2)*(1.94384**2)              # knots^2
    t2m['data'] = (t2m['data']-273.15)*1.8+32.                 # F
    slp['data'] = slp['data']/100.                           # hPa

    dus['data'] = dus['data']*1e9                            # ug/m3
    sss['data'] = sss['data']*1e9                            # ug/m3
    ocs['data'] = ocs['data']*1e9                            # ug/m3
    bcs['data'] = bcs['data']*1e9                            # ug/m3
    nis['data'] = nis['data']*1e9                            # ug/m3
    so4['data'] = so4['data']*1e9                            # ug/m3

    ###########################################################################

    for product in ['meteo', 'oc', 'bc', 'ss', 'du', 'su', 'total', 'co', 'co2', 'dumass', 'ssmass', 'ocmass', 'bcmass', 'sumass', 'nimass', 'totmass']:
        # start plot template
        fig = plt.figure()
        # ax = plt.gca()
        ax1 = plt.subplot2grid((30, 1), (1, 0))
        ax2 = plt.subplot2grid((30, 1), (2, 0), sharex=ax1)
        ax3 = plt.subplot2grid((30, 1), (3, 0), sharex=ax1)
        ax4 = plt.subplot2grid((30, 1), (4, 0), rowspan=15, sharex=ax1)
        ax5 = plt.subplot2grid((30, 1), (21, 0), rowspan=4)
        ax7 = plt.subplot2grid((30, 1), (26, 0), rowspan=4)

        # CLDHGH
        ax1.set_facecolor('b')
        ax1.bar(cldhgh_t, cldhgh['data'], width=width, color='w', align='center', linewidth=linewidth)
        ax1.bar(cldhgh_t, np.negative(cldhgh['data']), width=width, color='w', align='center', linewidth=linewidth)
        ax1.yaxis.set_ticks([])
        ax1.set_xlim([fcst, cldhgh_t[-1]])
        ax1.set_ylim(-1.5,1.5)
        ax1.xaxis.set_ticks_position('top')
        ax1.tick_params(axis=u'both', which=u'both', length=0)
        ax1.set_ylabel('High', rotation=0)
        ax1.yaxis.set_label_coords(-0.03, 0.1)

        # CLDMID
        ax2.set_facecolor('b')
        ax2.bar(cldhgh_t, cldmid['data'], width=width, color='w', align='center', linewidth=linewidth)
        ax2.bar(cldhgh_t, np.negative(cldmid['data']), width=width, color='w', align='center', linewidth=linewidth)
        ax2.set_ylim(-2,2)
        ax2.yaxis.set_ticks([])
        ax2.tick_params(axis=u'both', which=u'both', length=0)
        ax2.set_ylabel('Mid', rotation=0)
        ax2.yaxis.set_label_coords(-0.03, 0.1)
    
        # CLDLOW
        ax3.set_facecolor('b')
        ax3.bar(cldhgh_t, cldlow['data'], width=width, color='w', align='center', linewidth=linewidth)
        ax3.bar(cldhgh_t, np.negative(cldlow['data']), width=width, color='w', align='center', linewidth=linewidth)
        #ax3.axhline(0, color='b')
        ax3.yaxis.set_ticks([])
        ax3.tick_params(axis=u'both', which=u'both', length=0)
        ax3.set_ylim(-2,2)
        ax3.set_ylabel('Low', rotation=0)
        ax3.yaxis.set_label_coords(-0.03, 0.1)
    
        # skip ax4, ax5, ax6 till product loop (3d fields + slp/t2m)
        fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig.axes[1:3]+fig.axes[4:-1]], visible=False)
    
        # precip
        ax7.bar(precip_t, prectot['data'], width=width, color='g', align='center', linewidth=linewidth)
        ax7.bar(precip_t, precsno['data'], width=width*.6, color='b', align='center', linewidth=linewidth)
        ax7.bar(precip_t, preccon['data'], width=width*.4, color='r', align='center', linewidth=linewidth)
        ylow, yhigh = ax7.get_ylim()
        ax7.locator_params(axis='y', nbins=4)
        ax7.yaxis.grid(True, linestyle=':')
        ax7.yaxis.set_tick_params(direction='in')
        ax7.xaxis.set_tick_params(top=True,direction='in')
        ax7.set_ylim(ymin=0)
        ax7.xaxis.set_ticks([item for item in precip_t if item.hour == 0 or item.hour == 12])
        ax7.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(custom_date_formatter))
    
        # 2m winds
        ax8 = ax7.twinx()
        ax8.xaxis.set_ticks([item for item in precip_t if item.hour == 0 or item.hour == 12])
        ax8.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(custom_date_formatter))
        _2m_t = [fcst+dt.timedelta(seconds=int(item)) for item in u2m['time']]
        _2m = np.sqrt(u2m['data'] + v2m['data'])
        ax8.plot(_2m_t, _2m, color='k')
        ax8.set_xlim([fcst, precip_t[-1]])
        ax8.yaxis.set_tick_params(direction='in')
        ax8.set_ylim(ymin=0)
        ax8.locator_params(axis='y', nbins=4)
        del _2m_t
    
        # ax4, ax5, ax6 here
        ax4.xaxis.set_ticks([t for t in rh_t if t.hour == 0 or t.hour == 12])
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Hz'))

        # 3d field
        if product in 'meteo':
            ylabels = [int(rh['lev'][i]) for i in yi]
            ax4_img = ax4.contourf(x, y, rh['data'], cmap=cmap[product])
            CS = ax4.contour(x, y, rh['data'], colors='k', levels=[20,40,60,70,80,85,90,95])
        else:
            if product in 'total' or product in 'all':
                ylabels = [int(ocext['lev'][i]) for i in yi]
                _3d = ocext['data'] + bcext['data'] + ssext['data'] + duext['data'] + suext['data'] # + niext['data'].transose()[yi]
                ax4_img = ax4.contourf(x, y, _3d, cmap=cmap[product])
                CS = ax4.contour(x, y, _3d, colors='k')
            elif product in 'co':
                ylabels = [int(co['lev'][i]) for i in yi]
                _3d = co['data'] * airdens['data']
                ax4_img = ax4.contourf(x, y, _3d, cmap=cmap[product])
                CS = ax4.contour(x, y, _3d, colors='k')
            elif product in 'co2':
                ylabels = [int(co2['lev'][i]) for i in yi]
                _3d = co2['data'] * airdens['data']
                #ax4_img = ax4.contourf(x, y, _3d, cmap=cmap[product])
                #CS = ax4.contour(x, y, _3d, colors='k')
                ax4_img = ax4.imshow(_3d, cmap=cmap[product], origin='lower', interpolation=None, extent=[min(xi), max(xi), min(yi), max(yi)], aspect='auto')
            else:
                _3d = {
                    'oc':ocext, 'bc':bcext, 'ss':ssext, 'du':duext, 'su':suext, 'sumass':sumass, 'bcmass':bcmass, 'ocmass':ocmass, 'ssmass':ssmass, 'dumass':dumass, 'nimass':nimass, 'totmass':totmass
                }
                ylabels = [int(_3d[product]['lev'][i]) for i in yi]
                if 'mass' in product:
                    field = _3d[product]['data']*airdens['data']
                    # note: 2/28/2018
                    # Logarithmic colorscale/normalization errors when concentration is zero
                    # set a minimum of 0.01
                    # ultimately, we could also normalize all fields to a set range: [0.01, 100]
                    field_min = min(field.min(), 0.01)
                    field[field <= field_min] = field_min
                    if product in ['dumass', 'ocmass', 'totmass']:
             #          ax4_img = ax4.imshow(field, cmap=cmap[product], origin='lower', interpolation=None, extent=[min(xi), max(xi), min(yi), max(yi)], aspect='auto', norm=mcolors.LogNorm(vmin=0.01, vmax=500.))
                        ax4_img = ax4.imshow(field, cmap=cmap[product], origin='lower', interpolation=None, extent=[min(xi), max(xi), min(yi), max(yi)], aspect='auto', norm=mcolors.LogNorm(vmin=1.0, vmax=500.))
                    else:
                        ax4_img = ax4.imshow(field, cmap=cmap[product], origin='lower', interpolation=None, extent=[min(xi), max(xi), min(yi), max(yi)], aspect='auto', norm=mcolors.LogNorm(vmin=0.01, vmax=100.))
                    #ax4_img = ax4.imshow(field, cmap=cmap[product], origin='lower', interpolation=None, extent=[min(xi), max(xi), min(yi), max(yi)], aspect='auto', norm=mcolors.LogNorm(vmin=max(field.min(), 0.01), vmax=max(field.max(), 100))) 
                else:
                    ax4_img = ax4.contourf(x, y, _3d[product]['data'], cmap=cmap[product])
                    CS = ax4.contour(x, y, _3d[product]['data'], colors='k')
        fig.subplots_adjust(left=0.08, right=0.95)
        cbar_ax = fig.add_axes([0.965, 0.41, 0.02, 0.422])
        if 'mass' in product:
            cbar = fig.colorbar(ax4_img, cax=cbar_ax, orientation='vertical', format='%g', ticks=mpl.ticker.LogLocator(subs=[0,1,2.5,5]))
        else:
            cbar = fig.colorbar(ax4_img, cax=cbar_ax, orientation='vertical', format='%g')
            #cbar.formatter.set_powerlimits((0,0))
            #cbar.ax.yaxis.set_offset_position('left')
            #cbar.update_ticks()
        try:
            plt.setp(CS.collections, linewidth=0.2)
            plt.setp(CS.collections, linestyle='--')
            tl = ax4.clabel(CS, CS.levels, inline=1, fontsize=8, fmt='%g')
            for te in tl:
                te.set_bbox(dict(color='w', alpha=0.8, pad=0.1))
        except:
            pass
        ticks = ax4.get_yticks()          # returns current y-tick locations
        ax4.set_yticks(ticks[:-1])        # example: keep every other tick
        ax4.set_yticklabels(ylabels[::2][:-1])
        ax4.yaxis.set_tick_params(direction='out')
        ax4.yaxis.tick_left()
        ax4.xaxis.set_tick_params(direction='out')
        ax4.xaxis.tick_bottom()
        ax4.set_ylim(yi[1], yi[-1])

        # u/v winds
        ax4.barbs(
            x, y, u['data'], v['data'],
            barb_increments=dict(half=5, full=10, flag=50),
            linewidth=0.5
        )

        # slp & t2m
        if product in 'meteo':
            slp_t = [fcst+dt.timedelta(seconds=int(item)) for item in slp['time']]
            ax5.set_xlim([slp_t[0], slp_t[-1]])
            ax5.plot(slp_t, slp['data'], color='k')
            ax5.yaxis.get_major_formatter().set_useOffset(False)
            ax5.yaxis.grid(True, linestyle=':')
            ax5.tick_params(axis=u'both', which=u'both', length=0)
            ax5.locator_params(axis='y', nbins=3)
            ax6 = ax5.twinx()
            ax6.plot(slp_t, t2m['data'], color='r')
            ax6.yaxis.get_major_formatter().set_useOffset(False)
            ax6.yaxis.set_tick_params(direction='in')
            ax6.locator_params(axis='y', nbins=3)
            for t1 in ax6.get_yticklabels():
                t1.set_color('r')
        elif product in ['co', 'co2']:
            tau_t = [fcst+dt.timedelta(seconds=int(item)) for item in co['time']]
            tau_c = [fcst+dt.timedelta(seconds=int(item)) for item in cosc['time']]
            _co = {'co':co, 'co2':co2}
            _co = _co[product]['data']*airdens['data']
            _co = _co.sum(axis=0)
            ax5.set_xlim([tau_t[0], tau_t[-1]])
            ax5.bar(tau_t, _co, color='g', width=width, align='center', linewidth=linewidth, alpha=0.5)
            ax5.yaxis.grid(True, linestyle=':')
            ax5.set_ylim(auto=True)
            ticks = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1e3))
            ax5.yaxis.set_major_formatter(ticks)
            ax5.tick_params(axis=u'both', which=u'both', length=0)
            ax5.locator_params(axis='y', nbins=4)

            if product not in ['co2']:
                ax6 = ax5.twinx()
                ax6.plot(tau_c, cosc['data'], color='b')
                ax6.locator_params(axis='y', nbins=4)
                for t1 in ax6.get_yticklabels():
                    t1.set_color('b')
        elif 'mass' not in product:
            tau_t = [fcst+dt.timedelta(seconds=int(item)) for item in suexttau['time']]
            _su = suexttau['data']
            _c = _su + bcexttau['data'] + ocexttau['data']
            _du = _c + duexttau['data']
            _ss = _du + ssexttau['data']
            _ni = _ss +  niexttau['data'] # add nitrates
            ax5.set_xlim([tau_t[0], tau_t[-1]])
            ax5.bar(tau_t, _ni, color='b', width=width*.2, align='center', linewidth=linewidth)
            ax5.bar(tau_t, _ss, color='b', width=width*.2, align='center', linewidth=linewidth)
            ax5.bar(tau_t, _du, color='r', width=width*.2, align='center', linewidth=linewidth)
            ax5.bar(tau_t, _c, color='k', width=width*.2, align='center', linewidth=linewidth)
            ax5.bar(tau_t, _su, color='y', width=width*.2, align='center', linewidth=linewidth)
            ax5.yaxis.grid(True, linestyle=':')
            ax5.tick_params(axis=u'both', which=u'both', length=0)
            ax5.locator_params(axis='y', nbins=4)
        else:
            tau_t = tau_t = [fcst+dt.timedelta(seconds=int(item)) for item in so4['time']]
            _su = so4['data']
            _c = _su + bcs['data'] + ocs['data']
            _du = _c + dus['data']
            _ss = _du + sss['data']
            _ni = _ss + nis['data']
            ax5.set_xlim([tau_t[0], tau_t[-1]])
            ax5.bar(tau_t, _ni, color='g', width=width*.2, align='center', linewidth=linewidth)
            ax5.bar(tau_t, _ss, color='b', width=width*.2, align='center', linewidth=linewidth)
            ax5.bar(tau_t, _du, color='r', width=width*.2, align='center', linewidth=linewidth)
            ax5.bar(tau_t, _c, color='k', width=width*.2, align='center', linewidth=linewidth)
            ax5.bar(tau_t, _su, color='y', width=width*.2, align='center', linewidth=linewidth)
            ax5.yaxis.grid(True, linestyle=':')
            ax5.tick_params(axis=u'both', which=u'both', length=0)
            ax5.locator_params(axis='y', nbins=4)

        # Text Labels

        te = fig.text(0., 0.89, 'Clouds (%)', va='center', rotation='vertical', fontsize=11)

        te = fig.text(0., 0.6, 'Wind Barbs (knots)', va='center', rotation='vertical', fontsize=11)
        te_ = fig.text(-0.05, 0.15, 'Total Rain', va='center', rotation='vertical', fontsize=11, color='g')
        te = fig.text(-0.03, 0.15, 'Snow (Liq)', va='center', rotation='vertical', fontsize=11, color='b')
        te = fig.text(-0.01, 0.15, 'Convective', va='center', rotation='vertical', fontsize=11, color='r')
        te = fig.text(0.01, 0.15, '(mm)', va='center', rotation='vertical', fontsize=11)
        te_1 = fig.text(1.01, 0.15, '2m w spd', va='center', rotation=-90, fontsize=11)
        te = fig.text(0.99, 0.15, '(knots)', va='center', rotation=-90, fontsize=11)

        if product in 'meteo':
            te = fig.text(-0.03, 0.6, 'Relative Humidity (%)', va='center', rotation='vertical', fontsize=11, fontweight='bold')
            te = fig.text(-0.01, 0.3, 'T2M (F)', va='center', rotation='vertical', fontsize=11, color='r')
            te = fig.text(-0.04, 0.3, 'SLP (hPa)', va='center', rotation='vertical', fontsize=11)
        else:
            if product in 'su':
                te = fig.text(-0.03, 0.6, 'Sulfate Extinction (1/km)', va='center', rotation='vertical', fontsize=11, fontweight='bold')
            elif product in 'total' or product in 'all':
                te = fig.text(-0.03, 0.6, 'Total Aerosol Extinction (1/km)', va='center', rotation='vertical', fontsize=11, fontweight='bold')
            elif product in 'co':
                te = fig.text(-0.03, 0.6, r'CO concentration ($\mu$g/m$^3$)', va='center', rotation='vertical', fontsize=11, fontweight='bold')
            elif product in 'co2':
                te = fig.text(-0.03, 0.6, r'CO$_2$ concentration ($\mu$g/m$^3$)', va='center', rotation='vertical', fontsize=11, fontweight='bold')
            else:
                if 'totmass' in product:
                    te = fig.text(-0.03, 0.6, r' TOTAL PM ($\mu$g/m$^3$)', va='center', rotation='vertical', fontsize=11, fontweight='bold')
                elif 'mass' in product:
                    te = fig.text(-0.03, 0.6, product[:-4].upper() + r' PM2.5 ($\mu$g/m$^3$)', va='center', rotation='vertical', fontsize=11, fontweight='bold')
                else:
                    te = fig.text(-0.03, 0.6, product.upper()+' Extinction (1/km)', va='center', rotation='vertical', fontsize=11, fontweight='bold')
            if 'mass' in product:
                te_ = fig.text(-0.08, 0.3, 'Nitrate', va='center', rotation='vertical', fontsize=11, color='g')
                te = fig.text(-0.06, 0.3, 'Sea Salt', va='center', rotation='vertical', fontsize=11, color='b')
                te = fig.text(-0.04, 0.3, 'Dust', va='center', rotation='vertical', fontsize=11, color='r')
                te = fig.text(-0.02, 0.3, 'OC + BC', va='center', rotation='vertical', fontsize=11, color='k')
                te = fig.text(0., 0.3, 'Sulfate', va='center', rotation='vertical', fontsize=11, color='y')
                te = fig.text(0.02, 0.3, r'($\mu$g/m$^3$)', va='center', rotation='vertical', fontsize=11, color='k')
            elif product in ['co', 'co2']:
                te = fig.text(-0.02, 0.3, 'Total Column', va='center', rotation='vertical', fontsize=11, color='g')
                te = fig.text(0., 0.3, r'(mg/m$^3$)', va='center', rotation='vertical', fontsize=11)
                if product not in ['co2']:
                    te = fig.text(1., 0.3, '(ppbv)', va='center', rotation=-90, fontsize=11)
                    te = fig.text(1.02, 0.3, 'Surface CO', va='center', rotation=-90, fontsize=11)
            else:
                te_ = fig.text(-0.06, 0.3, 'Nitrate', va='center', rotation='vertical', fontsize=11, color='g')
                te = fig.text(-0.04, 0.3, 'Sea Salt', va='center', rotation='vertical', fontsize=11, color='b')
                te = fig.text(-0.02, 0.3, 'Dust', va='center', rotation='vertical', fontsize=11, color='r')
                te = fig.text(0.0, 0.3, 'OC + BC', va='center', rotation='vertical', fontsize=11, color='k')
                te = fig.text(0.02, 0.3, 'Sulfate', va='center', rotation='vertical', fontsize=11, color='y')
        lbl = fig.text(0.5,0.01, 'Lat = %.2f, Lon = %.2f, Location = %s, Fcst_Init = %s' % (float(lat), float(lon), station, forecast), ha='center', fontsize=11)

        # logos
        if 'mass' in product:
            ax = fig.add_axes([0, 1, 1, 0.05], anchor='NE', zorder=-1)
            ax.axis('off')

        img = '_'.join([product, str(lat), str(lon)])+'.png'
        plt.savefig(os.path.join(opath, img), bbox_inches='tight', dpi=100, bbox_extra_artists=(lbl,te_,te_1,te))
        print('Saving...'+img)

        plt.close()

        # logos
        if 'mass' in product:
            gmao_logo = '/discover/nobackup/dao_ops/ebsmith2/gram/GMAO-logo_small.png'
            nasa_logo = '/discover/nobackup/dao_ops/ebsmith2/gram/nasa-logo_small.png'
            os.system('composite -gravity northeast -geometry +35+30 ' + gmao_logo + ' ' + os.path.join(opath, img) + ' ' + os.path.join(opath, img))
            os.system('composite -gravity northwest -geometry +40+15 ' + nasa_logo + ' ' + os.path.join(opath, img) + ' ' + os.path.join(opath, img))

    del cldhgh, cldmid, cldlow, cldhgh_t
    del u, v, slp, t2m
    del rh, ocext, bcext, suext, co, co2, airdens, ssext, duext
    del ssexttau, duexttau, bcexttau, ocexttau, suexttau
    del sss, dus, bcs, ocs, so4
    del ssmass, dumass, ocmass, bcmass, sumass, nimass, totmass
    del prectot, precsno, preccon, u2m, v2m, precip_t

def get_data(ipath, field, station):
    try:
        # cf datagrams
        #if field in []:
        #    ipath = '/discover/nobackup/projects/gmao/geos_cf/pub/GEOS-CF_NRT_P///forecast/'
        fi = [x for x in find.find(path=ipath, ext='.nc') if '_'+field+'.' in x]
        if fi:
            fi = fi[0]
            d = Dataset(fi, 'r')

            # get station index
            st = [(i,x) for i,x in enumerate(d.ncattrs()) if 'Station' in x]
            s = [x for i,x in st if getattr(d, x) in station][0]
            s = int(s.split('_')[-1]) - 1

            if 'time' not in d.variables:
                return None
            # have to slice instead of obtaining pointer to netCDF4.Variable
            time = d.variables['time'][:]

            if 'lev' in d.variables:
                lev = d.variables['lev'][:]
            else:
                lev = None

            if field not in d.variables:
                return None
            data = d.variables[field][s]
            d.close()
            del d, s, st, fi
            return {'name':field, 'data':data, 'time':time, 'lev':lev}
    except:
        return None
