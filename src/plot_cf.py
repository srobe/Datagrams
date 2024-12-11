import os
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
#from matplotlib.colors import LinearSegmentedColormap as lsc
#m = lsc.from_list('name', ['white', 'saddlebrown'])
#m = lsc.from_list('name', ['white', 'magenta'])

#import contour as mcontour

# custom colormap for O3
purple01 = {'red': ((0.000, 0.992, 0.992),
                   (0.100, 0.976, 0.976),
                   (0.200, 0.945, 0.945),
                   (0.300, 0.906, 0.906),
                   (0.400, 0.847, 0.847),
                   (0.500, 0.773, 0.773),
                   (0.600, 0.682, 0.682),
                   (0.700, 0.576, 0.576),
                   (0.800, 0.455, 0.455),
                   (0.900, 0.318, 0.318),
                   (1.000, 0.165, 0.165)),
         'green': ((0.000, 0.953, 0.953),
                   (0.100, 0.843, 0.843),
                   (0.200, 0.690, 0.690),
                   (0.300, 0.525, 0.525),
                   (0.400, 0.369, 0.369),
                   (0.500, 0.243, 0.243),
                   (0.600, 0.149, 0.149),
                   (0.700, 0.086, 0.086),
                   (0.800, 0.047, 0.047),
                   (0.900, 0.008, 0.008),
                   (1.000, 0.000, 0.000)),
         'blue':  ((0.000, 0.992, 0.992),
                   (0.100, 0.976, 0.976),
                   (0.200, 0.945, 0.945),
                   (0.300, 0.906, 0.906),
                   (0.400, 0.847, 0.847),
                   (0.500, 0.773, 0.773),
                   (0.600, 0.682, 0.682),
                   (0.700, 0.576, 0.576),
                   (0.800, 0.455, 0.455),
                   (0.900, 0.318, 0.318),
                   (1.000, 0.165, 0.165))
}
hot01 = {
    'red':   ((0.000, 1.000, 1.000),
              (0.250, 1.000, 1.000),
              (0.500, 1.000, 1.000),
              (0.750, 0.886, 0.886),
              (1.000, 0.471, 0.471)),
    'green': ((0.000, 0.973, 0.973),
              (0.250, 0.757, 0.757),
              (0.500, 0.380, 0.380),
              (0.750, 0.067, 0.067),
              (1.000, 0.000, 0.000)),
    'blue':  ((0.000, 0.612, 0.612),
              (0.250, 0.259, 0.259),
              (0.500, 0.106, 0.106),
              (0.750, 0.055, 0.055),
              (1.000, 0.063, 0.063))
}
green01 = {
    'red':   ((0.000, 0.867, 0.867),
              (0.100, 0.729, 0.729),
              (0.200, 0.576, 0.576),
              (0.300, 0.427, 0.427),
              (0.400, 0.298, 0.298),
              (0.500, 0.196, 0.196),
              (0.600, 0.125, 0.125),
              (0.700, 0.075, 0.075),
              (0.800, 0.039, 0.039),
              (0.900, 0.012, 0.012),
              (1.000, 0.000, 0.000)),
    'green': ((0.000, 0.976, 0.976),
              (0.100, 0.953, 0.953),
              (0.200, 0.918, 0.918),
              (0.300, 0.871, 0.871),
              (0.400, 0.808, 0.808),
              (0.500, 0.733, 0.733),
              (0.600, 0.643, 0.643),
              (0.700, 0.541, 0.541),
              (0.800, 0.424, 0.424),
              (0.900, 0.294, 0.294),
              (1.000, 0.157, 0.157)),
    'blue':  ((0.000, 0.867, 0.867),
              (0.100, 0.729, 0.729),
              (0.200, 0.576, 0.576),
              (0.300, 0.427, 0.427),
              (0.400, 0.298, 0.298),
              (0.500, 0.196, 0.196),
              (0.600, 0.122, 0.122),
              (0.700, 0.075, 0.075),
              (0.800, 0.035, 0.035),
              (0.900, 0.008, 0.008),
              (1.000, 0.000, 0.000))
}
comp02 = {
    'red':   ((0.000, 0.000, 0.000),
              (0.333, 0.820, 0.820),
              (0.667, 0.847, 0.847),
              (1.000, 0.667, 0.667)),
    'green': ((0.000, 1.000, 1.000),
              (0.333, 0.800, 0.800),
              (0.667, 0.514, 0.514),
              (1.000, 0.216, 0.216)),
    'blue':  ((0.000, 1.000, 1.000),
              (0.333, 0.259, 0.259),
              (0.667, 0.184, 0.184),
              (1.000, 0.098, 0.098))
}
#==============================================================================

purple01 = mcolors.LinearSegmentedColormap('purple01', purple01)
plt.register_cmap(cmap=purple01)

hot01 = mcolors.LinearSegmentedColormap('hot01', hot01)
plt.register_cmap(cmap=hot01)

green01 = mcolors.LinearSegmentedColormap('green01', green01)
plt.register_cmap(cmap=green01)

comp02 = mcolors.LinearSegmentedColormap('comp02', comp02)
plt.register_cmap(cmap=comp02)

#cmap = {
#    'o3':purple01,
#    'co':plt.cm.Reds,
#    'no2':plt.cm.Greens,
#    'so2':plt.cm.Wistia,
#    'hcho':plt.cm.viridis,
#    'pm25':plt.cm.Oranges,
#}

cmap = {
    'o3':purple01,
    'co':hot01,
    'no2':green01,
    'so2':plt.cm.Wistia,
    'hcho':plt.cm.viridis,
    'pm25':plt.cm.Oranges,
}

width = 0.118
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
        d = mdates.date2num(forecast + dt.timedelta(days=1))
        if x < d:
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

    #try:
    #    from pip._internal.operations import freeze
    #except ImportError:  # pip < 10.0
    #    from pip.operations import freeze

    #x = freeze.freeze()
    #for p in x:
    #    print(p)

    #import time
    #start = time.time()
    # parallelize across stations
    #pool = mp.Pool(processes=8)
    #pool.map(plot_wrapper, s)
    #pool.close()
    #pool.join()
    #print 'Elapsed Time: %.2fs' % ((time.time()-start))

    # single processor
    #plot_wrapper(*s)

    # multiprocessing
    pool = mp.Pool(processes=8)
    pool.map(plot_wrapper, s)
    pool.close()
    pool.join()

    return

def plot_wrapper(args):
    return plot(*args)


#############################################################################

def plot(ipath, station, lat, lon, fcst, opath):
    '''main plot driver'''
    global forecast
    forecast = fcst
    # plot settings
    rc()

    # shared by all plots
    cldtt = get_data(ipath, 'CLDTT', station) # clouds
    o3 = get_data(ipath, 'O3', station, collection='chm_inst_1hr_g1440x721_p23')
    so2 = get_data(ipath, 'SO2', station, collection='chm_inst_1hr_g1440x721_p23')
    co = get_data(ipath, 'CO', station, collection='chm_inst_1hr_g1440x721_p23')
    no2 = get_data(ipath, 'NO2', station, collection='chm_inst_1hr_g1440x721_p23')
    #hcho = get_data(ipath, 'HCHO', station)
    pm25 = get_data(ipath, 'PM25_RH35_GCC', station, collection='chm_inst_1hr_g1440x721_p23')

    precip = get_data(ipath, 'TPREC', station)
    t2m = get_data(ipath, 'T2M', station)

    # surface fields
    o3_sfc = get_data(ipath, 'O3', station, collection='chm_tavg_1hr_g1440x721_v1')
    so2_sfc = get_data(ipath, 'SO2', station, collection='chm_tavg_1hr_g1440x721_v1')
    co_sfc = get_data(ipath, 'CO', station, collection='chm_tavg_1hr_g1440x721_v1')
    no2_sfc = get_data(ipath, 'NO2', station, collection='chm_tavg_1hr_g1440x721_v1')

    u = get_data(ipath, 'U10M', station)
    v = get_data(ipath, 'V10M', station)

    ni = get_data(ipath, 'PM25ni_RH35_GCC', station)
    ss = get_data(ipath, 'PM25ss_RH35_GCC', station)
    du = get_data(ipath, 'PM25du_RH35_GCC', station)
    oc = get_data(ipath, 'PM25oc_RH35_GCC', station)
    bc = get_data(ipath, 'PM25bc_RH35_GCC', station)
    su = get_data(ipath, 'PM25su_RH35_GCC', station)
    org = get_data(ipath, 'PM25soa_RH35_GCC', station)

    pblh = get_data(ipath, 'ZPBL', station)

    ############################################################################

    ## pressure level axis
    yi = [i for i in range(len(o3['lev'])) if 500. <= o3['lev'][i] <= 1000.]

    ## time axes
    cld_t = [fcst+dt.timedelta(seconds=int(item)) for item in cldtt['time']]
    #precip_t = [fcst+dt.timedelta(seconds=int(item)) for item in prectot['time']]

    o3_t = [fcst+dt.timedelta(seconds=int(t)) for t in o3['time']]
    x = np.arange(len(o3_t))
    x = [fcst+dt.timedelta(hours=int(ti)) for ti in x] # hourly data!
    #x = mdates.date2num(x)
    #xi = x
    xi = mdates.date2num(x)
    x, y = np.meshgrid(x, yi)

    ############################################################################

    ## scaling for imagery
    cldtt['data'] = np.absolute(cldtt['data'])                # %
    precip['data'] = precip['data']*1e4                       # mm

    o3['data'] = o3['data'].transpose()[yi]*1.0e+9            # PPBV
    so2['data'] = so2['data'].transpose()[yi]*1.0e+9          # PPBV
    co['data'] = co['data'].transpose()[yi]*1.0e+9            # PPBV
    no2['data'] = no2['data'].transpose()[yi]*1.0e+9          # PPBV
    #hcho['data'] = hcho['data'].transpose()[yi]               # ?
    pm25['data'] = pm25['data'].transpose()[yi]               # ug m-3

    o3_sfc['data'] = o3_sfc['data']*1.0e+9
    co_sfc['data'] = co_sfc['data']*1.0e+9
    no2_sfc['data'] = no2_sfc['data']*1.0e+9
    so2_sfc['data'] = so2_sfc['data']*1.0e+9
    
    wspeed = np.sqrt(u['data']**2 + v['data']**2)
    t2m['data'] = (t2m['data']-273.15)*1.8+32.                 # F

    ############################################################################

    for product in ['o3', 'so2', 'co', 'no2', 'pm25']: # iterate per product
        # start plot template
        fig = plt.figure()
        ax = plt.gca()
        ax1 = plt.subplot2grid((28, 1), (1, 0))
        #ax2 = plt.subplot2grid((28, 1), (2, 0), sharex=ax1)
        #ax3 = plt.subplot2grid((28, 1), (3, 0), sharex=ax1)
        ax4 = plt.subplot2grid((28, 1), (2, 0), rowspan=15, sharex=ax1)
        ax5 = plt.subplot2grid((28, 1), (19, 0), rowspan=4)
        ax7 = plt.subplot2grid((28, 1), (24, 0), rowspan=4)

        # CLDTT
        ax1.set_facecolor('b')
        #ax1.set_facecolor('b')
        ax1.bar(cld_t[::3], cldtt['data'][::3], width=width, color='w', align='center', linewidth=linewidth)
        #ax1.bar(cld_t[::3], np.negative(cldtt['data'][::3]), width=width, color='w', align='center', linewidth=linewidth)
        ax1.yaxis.set_ticks([])
        ax1.set_xlim([fcst, cld_t[-1]])
        ax1.set_ylim(0,1.5)
        ax1.xaxis.set_ticks_position('top')
        ax1.tick_params(axis=u'both', which=u'both', length=0)
        ax1.set_ylabel('Tot Cld (%)', rotation=0)
        ax1.yaxis.set_label_coords(-0.07, 0.1)

        # skip ax4, ax5, ax6 till product loop (3d fields + slp/t2m)
        fig.subplots_adjust(hspace=0)
        # blank x-axis
        plt.setp([a.get_xticklabels() for a in fig.axes[2:-1]], visible=False)
    
        # ax4, ax5, ax6 here
        ax4.xaxis.set_ticks([t for t in o3_t if t.hour == 0 or t.hour == 12])
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Hz'))

        # 3d fields
        _3d = {
            'o3':o3, 'so2':so2, 'co':co, 'pm25':pm25,'no2':no2,
        }
        limits = {
            'o3':(0.01,100), 'so2':(0.01,40), 'co':(0.01,250), 'no2':(0.01,100), 'pm25':(0.01,150),
        }
        scale = {
            'o3':range(10,111,5),
            'so2':[0.01, 0.02, 0.03, 0.04, 0.07, 0.1, 0.2, 0.3, 0.5, 0.8, 1.3, 2, 4, 5, 8, 10, 15, 20, 50, 100],
            'co':range(30,256,15),
            'no2':[0.01, 0.02, 0.03, 0.04, 0.07, 0.1, 0.2, 0.3, 0.5, 0.8, 1.3, 2, 3.5, 5, 10, 15, 25, 40, 60, 100],
            'pm25':[1, 2, 3, 5, 8, 10, 20, 25, 30, 40, 50, 60, 80, 90, 100, 150, 200, 300, 400, 500],
        }
        field = _3d[product]['data']
        ylabels = [int(_3d[product]['lev'][i]) for i in yi]
        #field_min = min(field.min(), limits[product][0])
        #field[field <= field_min] = field_min
        #field_max = max(field.max(), limits[product][1])

        #v = np.logspace(np.log10(limits[product][0]), np.log10(limits[product][1]), 20)
        v = np.linspace(limits[product][0], limits[product][1], 20)
        mcmap = cmap[product]
        mcmap.set_bad('silver', 1.)
        mcmap.set_under('w')
        mcmap.set_over('k')

        # didn't need to change the field minimum at all
        #axis_log = [0.0, 0.63, 1.28, 1.95, 2.63, 4.06, 5.57, 7.19, 8.91, 10.76, 12.76, 14.93, 17.31]


        if 'co' in product or 'o3' in product:
            ax4_img = ax4.contourf(x,y, field, levels=scale[product], cmap=mcmap, extend='both')
        else:
            # coerce the bad values to be within the range
            #if field.min() < min(scale[product]):
            #    #field[field <= min(scale[product])] = min(scale[product])
            #    field = np.where(field <= min(scale[product]), min(scale[product]), field)
            #if field.max() > max(scale[product]):
            #    field = np.where(field >= max(scale[product]), max(scale[product]), field)
            #scale[product] = [np.log10(i) for i in scale[product]]
            mcmap = cmap[product]  # define the colormap
            # extract all colors from the .jet map
            cmaplist = [mcmap(i) for i in range(mcmap.N)]
            # force the first color entry to be grey
            #cmaplist[0] = (.5, .5, .5, 1.0)
            #cmaplist[0] = (1., 1., 1., 1.0)
            #cmaplist[-1] = (0., 0., 0., 1.)
            
            # create the new map
            mcmap = mpl.colors.LinearSegmentedColormap.from_list(
                'Custom cmap', cmaplist, mcmap.N)
            mcmap.set_under('w')
            mcmap.set_over('k')
            
            # define the bins and normalize
            #bounds = np.linspace(0, 20, 21)
            bounds = scale[product]
            norm = mpl.colors.BoundaryNorm(bounds, mcmap.N)
            ax4_img = ax4.contourf(x,y, field, levels=scale[product], cmap=mcmap, norm=norm, extend='both') #norm=mcolors.LogNorm(vmin=min(scale[product]), vmax=max(scale[product]), clip=False))
            #print(mpl.rcParams['image.interpolation'])
	    #ax4_img = ax4.imshow(field, cmap=mcmap, origin='lower', interpolation=None, extent=[min(xi), max(xi), min(yi), max(yi)], aspect='auto', norm=mcolors.LogNorm(vmin=min(scale[product]), vmax=max(scale[product]), clip=False))
        CS = ax4.contour(x,y,field, colors='k', levels=scale[product]) #[0.01, 1, 5, 15, 40, 70, 100, 120])
#        ax4_img.set_clim(max(field.min(), min(scale[product])), min(max(scale[product]), field.max()))
        ax4.patch.set_facecolor('silver')
        #ax4.set_yscale('log')
        #if 'so2' in product:
        #    print(field.min(), field.max(), float(lat), float(lon))

        fig.subplots_adjust(left=0.08, right=0.95)
        cbar_ax = fig.add_axes([0.965, 0.437, 0.02, 0.449])
        if product in ['no2', 'so2']:
            cbar = fig.colorbar(ax4_img, cax=cbar_ax, orientation='vertical', format='%7.2f', ticks=scale[product]) #, boundaries=scale[product]) #, extend='both')
            cbar.ax.yaxis.set_tick_params(pad=40,direction='in')
        else:
            cbar = fig.colorbar(ax4_img, cax=cbar_ax, orientation='vertical', format='%g', ticks=scale[product])
            cbar.ax.yaxis.set_tick_params(pad=25,direction='in')

        labels = {
           'o3':'O$_3$ (PPBV)',
            'so2':'SO$_2$ (PPBV)',
            'no2':'NO$_2$ (PPBV)',
            'co':'CO (PPBV)',
#            'hcho':'HCHO',
            'pm25':'PM2.5 ($\mu$g/m$^3$)',
        }

        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),ha='right')
        cbar.set_label(labels[product], fontsize=11, fontweight='bold', rotation=270, labelpad=25)
        try:
            plt.setp(CS.collections, linewidth=0.4)
            plt.setp(CS.collections, linestyle='-')
            if product in ['no2', 'so2']:
                tl = ax4.clabel(CS, CS.levels, inline=1, fontsize=8, fmt='%.2f')
            else:
                tl = ax4.clabel(CS, CS.levels, inline=1, fontsize=8, fmt='%d')
            for te in tl:
                te.set_bbox(dict(color='w', alpha=0.8, pad=0.1))
        except:
            pass
        ax4.set_yticklabels(ylabels[::2][:-1])
        ax4.yaxis.set_tick_params(direction='out')
        ax4.yaxis.tick_left()
        ax4.xaxis.set_tick_params(direction='out')
        ax4.xaxis.tick_bottom()
        #ax4.set_ylim(yi[1], yi[-1])
        ax4.set_ylabel('Pressure (hPa)', fontsize=11)

#        ######################################################################
#        # mimic pressure level axis, but change to heights via hypsometric equation
#        # Emma's code:
#        #    ylabels=[0,2,4,6,8,10,12]
#        #    yvals2=EXP(-1.0*(ylabels/7.4-ALOG(1000.0)))
#        #    AXES, /onlyright, yvals=yvals2, ylabels=ylabels, ytitle='Height (km)'
#        # note: this is only an estimate based on the static pressure levels
#        #       eventually this would be dynamically varying
#
#        axh = ax4.twinx()
#        #heights_f = lambda hPa: np.exp(-1.*(hPa/7.4-np.log(1000.0)))
#        heights_f = lambda hPa: 7.4*(np.log(1000.0)-np.log(hPa))
#        # get left axis limits
#        locs, labs = ax4.get_yticks(), ax4.get_yticklabels()
#        # apply function and set transformed values to right axis limits
#
#        # set an invisible artist to twin axes
#        # to prevent falling back to initial values on rescale events
#        axh.set_ylim((heights_f(ylabels[int(locs[0])]), heights_f(ylabels[int(locs[-1])])))
#        axh.set_yticks(locs)
#        axh.set_yticklabels(['%.2f' % heights_f(ylabels[int(lab)]) for lab in locs])
#        axh.spines['left'].set_position(('axes', -0.15))
#        axh.set_frame_on(True)
#        axh.patch.set_visible(False)
#        for sp in axh.spines.values():
#            sp.set_visible(False)
#        axh.spines['left'].set_visible(True)
#        axh.plot([i+dt.timedelta(minutes=30) for i in cld_t], pblh['data']/1000., 'r', linewidth=1.2)
#        axh.yaxis.set_label_position('left')
#        axh.yaxis.set_ticks_position('left')
#        axh.set_ylabel('Approx. Height (km)', labelpad=15, fontsize=11)
#
#        ######################################################################

        if 'pm25' not in product:
            sfc = {
                'o3':o3_sfc, 'so2':so2_sfc, 'co':co_sfc, 'no2':no2_sfc,
            }
            sfc_t = [fcst+dt.timedelta(seconds=int(item)) for item in sfc[product]['time']]
            ax5.set_xlim([sfc_t[0], sfc_t[-1]])
            ax5.plot(sfc_t, sfc[product]['data'], 'k')
            #if 'so2' in product:
                #print(field.shape)
                #print(len(sfc_t))
                #print(sfc[product]['data'].shape)
                #or i in range(len( _3d[product]['data'])):
                #ax5.plot(o3_t, _3d[product]['data'][1], 'r', linewidth=1.5)
                #ax5.plot(o3_t, _3d[product]['data'][0], 'r', linewidth=1.5)
                #print(_3d[product]['data'][0].min(), _3d[product]['data'][0].max())
                #print(_3d[product]['lev'][1])
            #ax5.plot(sfc_t, field[0], 'r')
            ax5.yaxis.set_tick_params(direction='in')
            ax5.xaxis.set_tick_params(top=True,direction='in')
            ax5.locator_params(axis='y', nbins=3, min_n_ticks=4)
            ax5.yaxis.grid(True,linestyle=':')
            ax5.xaxis.set_ticks([item for item in cld_t if item.hour == 0 or item.hour == 12])
            ax5.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(custom_date_formatter))
            ax5.set_ylabel('Surface '+labels[product].replace(labels[product].split(' ')[-1],'')[:-1] + '\n(PPBV)', fontsize=11)
        else:
            tau_t = [fcst+dt.timedelta(seconds=int(item)) for item in org['time']]
            ax5.set_xlim([tau_t[0], tau_t[-1]])
            _org = org['data']
            _su = su['data'] + _org
            _boc = bc['data'] + oc['data'] + _su
            _du = du['data'] + _boc
            _ss = ss['data'] + _du
            _ni = ni['data'] + _ss

            ax5.bar(tau_t, _ni.flatten(), color='g', width=width*.2, align='center', linewidth=linewidth)
            ax5.bar(tau_t, _ss.flatten(), color='b', width=width*.2, align='center', linewidth=linewidth)
            ax5.bar(tau_t, _du.flatten(), color='r', width=width*.2, align='center', linewidth=linewidth)
            ax5.bar(tau_t, _boc.flatten(), color='k', width=width*.2, align='center', linewidth=linewidth)
            ax5.bar(tau_t, _su.flatten(), color='y', width=width*.2, align='center', linewidth=linewidth)
            ax5.bar(tau_t, _org.flatten(), color='darkorange', width=width*.2, align='center', linewidth=linewidth)
            ax5.yaxis.grid(True,linestyle=':')
            ax5.yaxis.set_tick_params(direction='in')
            ax5.locator_params(axis='y', nbins=4)

            ax6 = ax5.twinx()
            ax6.yaxis.set_ticks([])
            ax6.set_ylabel('$\mu$g/m$^3$', color='k', rotation='-90', fontsize=11, labelpad=22)

        precip_t = [fcst+dt.timedelta(seconds=int(item)) for item in precip['time']]
        ax7.set_xlim([precip_t[0], precip_t[-1]])
        ax7.bar(precip_t, precip['data'], color='b', width=width*.5, align='center', linewidth=linewidth, alpha=0.5)
        ax7.yaxis.offsetText.set_color('b')
        ax7.yaxis.get_major_formatter().set_useOffset(False)
        ax7.yaxis.grid(True,linestyle=':')
        ax7.yaxis.set_tick_params(direction='in')
        ax7.xaxis.set_ticks([item for item in cld_t if item.hour == 0 or item.hour == 12])
        ax7.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(custom_date_formatter))
        ax8 = ax7.twinx()
        ax8.plot(precip_t, t2m['data'], color='r')
        ax8.yaxis.offsetText.set_color('r')
        ax8.yaxis.get_major_formatter().set_useOffset(False)
        ax7.tick_params(axis=u'both', which=u'both', length=0)
        ax7.locator_params(axis='y', nbins=4)
        ax8.locator_params(axis='y', nbins=4)
        ax8.yaxis.set_tick_params(direction='in')
        ax7.set_ylabel('Tot Precip\n(mm)', color='b', fontsize=11)
        ax8.set_ylabel('Temp 2m ($^\circ$F)', color='r', rotation='-90', fontsize=11, labelpad=15)
        for t1 in ax8.get_yticklabels():
            t1.set_color('r')
        for t1 in ax7.get_yticklabels():
            t1.set_color('b')
        # wind speed
        ax9 = ax7.twinx()
        ax9.spines['right'].set_position(('axes', 1.1))
        ax9.set_frame_on(True)
        ax9.patch.set_visible(False)
        for sp in ax9.spines.values():
            sp.set_visible(False)
        ax9.spines['right'].set_visible(True)

        ax9.plot(precip_t, wspeed, color='k')
        ax9.yaxis.get_major_formatter().set_useOffset(False)
        ax9.locator_params(axis='y', nbins=4)
        ax9.yaxis.set_tick_params(direction='in')
        ax9.set_ylabel('Wind 10m (m/s)', color='k', fontsize=11, rotation='-90', labelpad=15)
        ax9.xaxis.set_ticks([item for item in cld_t if item.hour == 0 or item.hour == 12])
        ax9.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(custom_date_formatter))

        if 'pm25' in product:
            te_ = fig.text(-0.08, 0.3, 'Nitrate', va='center', rotation='vertical', fontsize=11, color='g')
            te = fig.text(-0.06, 0.3, 'Sea Salt', va='center', rotation='vertical', fontsize=11, color='b')
            te = fig.text(-0.04, 0.3, 'Dust', va='center', rotation='vertical', fontsize=11, color='r')
            te = fig.text(-0.02, 0.3, 'OC + BC', va='center', rotation='vertical', fontsize=11, color='k')
            te = fig.text(0.0, 0.3, 'Sulfate', va='center', rotation='vertical', fontsize=11, color='y')
            te = fig.text(0.02, 0.3, 'SOA', va='center', rotation='vertical', fontsize=11, color='darkorange')
        fig.suptitle('GEOS CF Forecast Initialized on %s\n'% (forecast.strftime('%Hz %m/%d/%Y')), fontsize=14, fontweight='bold', y=1.02)
        lbl = fig.text(0.5,-0.01, 'Lat = %.2f, Lon = %.2f, Location = %s, Fcst_Init = %s' % (float(lat), float(lon), station, forecast), ha='center', fontsize=11, fontweight='bold')

        # logos
        ax = fig.add_axes([0, 1, 1, 0.05], anchor='NE', zorder=-1)
        ax.axis('off')

        img = '_'.join(['cf',product, str(lat), str(lon)])+'.png'
        plt.savefig(os.path.join(opath, img), bbox_inches='tight', dpi=100) #, bbox_extra_artists=(lbl,te_,te_1,te))
        print('Saving...'+img)

        plt.close()

        # logos
        #if 'mass' in product:
        gmao_logo = '/discover/nobackup/dao_ops/ebsmith2/gram/GMAO-logo_small.png'
        nasa_logo = '/discover/nobackup/dao_ops/ebsmith2/gram/nasa-logo_small.png'
        os.system('composite -gravity northeast -geometry +35+30 ' + gmao_logo + ' ' + os.path.join(opath, img) + ' ' + os.path.join(opath, img))
        os.system('composite -gravity northwest -geometry +40+15 ' + nasa_logo + ' ' + os.path.join(opath, img) + ' ' + os.path.join(opath, img))

    del cldtt
    #del cldhgh, cldmid, cldlow, cldhgh_t
    #del u, v, slp, t2m
    #del rh, ocext, bcext, suext, co, co2, airdens, ssext, duext
    #del ssexttau, duexttau, bcexttau, ocexttau, suexttau
    #del sss, dus, bcs, ocs, so4
    #del ssmass, dumass, ocmass, bcmass, sumass, nimass
    #del prectot, precsno, preccon, u2m, v2m, precip_t

def get_data(ipath, field, station, collection=None):
    try:
        # cf datagrams - same as the other since these are station sampled files
        #if field in []:
        #    ipath = '/discover/nobackup/projects/gmao/geos_cf/pub/GEOS-CF_NRT_P///forecast/'
        if collection:
            fi = [x for x in find.find(path=ipath, ext='.nc') if '_'+field+'.' in x and collection in x]
        else:
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
