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
    'co2':hot01,
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
    cldtot = get_data(ipath, 'CLDTOT', station) # clouds
    co2 = get_data(ipath, 'CO2', station, collection='inst3_3d_carb_Np')
    co2['data'] = np.flip(co2['data'], 1)
    co2['lev'] = np.flip(co2['lev'])

    precip = get_data(ipath, 'PRECTOT', station)
    t2m = get_data(ipath, 'T2M', station)
    xco2 = get_data(ipath, 'XCO2', station)

    # surface fields
    co2_sfc = get_data(ipath, 'CO2', station, collection='inst3_3d_carb_Nv')

    co2_sfc['data'] = co2_sfc['data'][:,-1]*1.0e+6
    u = get_data(ipath, 'U10M', station)
    v = get_data(ipath, 'V10M', station)

    ############################################################################

    ## pressure level axis
    yi = [i for i in range(len(co2['lev'])) if 500. <= co2['lev'][i] <= 1000.]

    ## time axes
    cld_t = [fcst+dt.timedelta(seconds=int(item)) for item in cldtot['time']]

    co2_t = [fcst+dt.timedelta(seconds=int(t)) for t in co2['time']]
    x = np.arange(len(co2_t))
    x = [fcst+dt.timedelta(hours=int(ti)*3) for ti in x] # hourly data!
    xi = mdates.date2num(x)
    x, y = np.meshgrid(x, yi)

    ############################################################################

    ## scaling for imagery
    cldtot['data'] = np.absolute(cldtot['data'])                # %
    precip['data'] = precip['data']*3600

    co2['data'] = co2['data'].transpose()[yi]*1.0e+6            # PPMV
    xco2['data'] = xco2['data'] * 1.0e+6                        # PPMV
  # co2_sfc['data'] = co2_sfc['data'].flatten()*1.0e+6

    co2max = round(co2['data'].max())
    co2min = int(co2['data'].min())
    co2int = (co2max - co2min) / 20.0

    while co2int < 1.0:

        co2max += 1.0
        co2int = (co2max - co2min) / 20.0

  # co2int = round(co2int*10.0) / 10.0
    co2int = round(co2int)

    co2val = co2min
    co2scale = []
    while co2val <= co2max:

        co2scale.append(co2val)
        co2val += co2int
    
    wspeed = np.sqrt(u['data']**2 + v['data']**2)
    t2m['data'] = (t2m['data']-273.15)*1.8+32.                 # F

    ############################################################################

    for product in ['co2']: # iterate per product
        # start plot template
        fig = plt.figure()
        ax = plt.gca()
        ax1 = plt.subplot2grid((28, 1), (1, 0))
        #ax2 = plt.subplot2grid((28, 1), (2, 0), sharex=ax1)
        #ax3 = plt.subplot2grid((28, 1), (3, 0), sharex=ax1)
        ax4 = plt.subplot2grid((28, 1), (2, 0), rowspan=15, sharex=ax1)
        ax5 = plt.subplot2grid((28, 1), (19, 0), rowspan=4)
        ax7 = plt.subplot2grid((28, 1), (24, 0), rowspan=4)

        # CLDTOT
        ax1.set_facecolor('b')
        ax1.bar(cld_t[::3], cldtot['data'][::3], width=width, color='w', align='center', linewidth=linewidth)
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
        ax4.xaxis.set_ticks([t for t in co2_t if t.hour == 0 or t.hour == 12])
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Hz'))

        # 3d fields
        _3d = {
            'co2':co2,
        }
        limits = {
            'co2':(0.01,500),
        }
        scale = {
         #  'co2':range(390,460,5),
         #  'co2': [400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 415, 420, 425, 430, 435, 440, 445, 450],
            'co2': co2scale
        }
        field = _3d[product]['data']
        ylabels = [int(_3d[product]['lev'][i]) for i in yi]

        v = np.linspace(limits[product][0], limits[product][1], 20)
        mcmap = cmap[product]
        mcmap.set_bad('silver', 1.)
        mcmap.set_under('w')
        mcmap.set_over('k')

        if 'co2' in product:
            ax4_img = ax4.contourf(x,y, field, levels=scale[product], cmap=mcmap, extend='both')
        else:
            mcmap = cmap[product]  # define the colormap
            # extract all colors from the .jet map
            cmaplist = [mcmap(i) for i in range(mcmap.N)]
            
            # create the new map
            mcmap = mpl.colors.LinearSegmentedColormap.from_list(
                'Custom cmap', cmaplist, mcmap.N)
            mcmap.set_under('w')
            mcmap.set_over('k')
            
            # define the bins and normalize
            #bounds = np.linspace(0, 20, 21)
            bounds = scale[product]
            norm = mpl.colors.BoundaryNorm(bounds, mcmap.N)
            ax4_img = ax4.contourf(x,y, field, levels=scale[product], cmap=mcmap, norm=norm, extend='both') 
        CS = ax4.contour(x,y,field, colors='k', levels=scale[product]) #[0.01, 1, 5, 15, 40, 70, 100, 120])
        ax4.patch.set_facecolor('silver')

        fig.subplots_adjust(left=0.08, right=0.95)
        cbar_ax = fig.add_axes([0.965, 0.437, 0.02, 0.449])

        cbar = fig.colorbar(ax4_img, cax=cbar_ax, orientation='vertical', format='%g', ticks=scale[product])
        cbar.ax.yaxis.set_tick_params(pad=25,direction='in')

        labels = {
            'co2':'CO$_2$ (PPMV)',
        }

        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),ha='right')
        cbar.set_label(labels[product], fontsize=11, fontweight='bold', rotation=270, labelpad=25)
        try:
            plt.setp(CS.collections, linewidth=0.4)
            plt.setp(CS.collections, linestyle='-')
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

        sfc = {
            'co2':co2_sfc,
        }
        sfc_t = [fcst+dt.timedelta(seconds=int(item)) for item in sfc[product]['time']]
        ax5.set_xlim([sfc_t[0], sfc_t[-1]])
        ax5.plot(sfc_t, sfc[product]['data'], 'k')
        ax5.yaxis.set_tick_params(direction='in')
        ax5.xaxis.set_tick_params(top=True,direction='in')
        ax5.locator_params(axis='y', nbins=3, min_n_ticks=4)
        ax5.yaxis.grid(True,linestyle=':')
        ax5.xaxis.set_ticks([item for item in cld_t if item.hour == 0 or item.hour == 12])
        ax5.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(custom_date_formatter))
        ax5.set_ylabel('Surface '+labels[product].replace(labels[product].split(' ')[-1],'')[:-1] + '\n(PPMV)', fontsize=11)

        ax5b = ax5.twinx()
        ax5b.plot(sfc_t, xco2['data'], 'r')
        ax5b.yaxis.offsetText.set_color('r')
        ax5b.yaxis.get_major_formatter().set_useOffset(False)
        ax5.tick_params(axis=u'both', which=u'both', length=0)
        ax5b.locator_params(axis='y', nbins=3, min_n_ticks=4)
        ax5b.yaxis.set_tick_params(direction='in')
        ax5b.set_ylabel('Column '+labels[product].replace(labels[product].split(' ')[-1],'')[:-1] + '\n(PPMV)', rotation='-90', fontsize=11, labelpad=30)
        for t1 in ax5b.get_yticklabels():
            t1.set_color('r')


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

        start_date = forecast.strftime('%Y%m%d')
        end_date = (forecast+dt.timedelta(days=5)).strftime('%Y%m%d')
        fig.suptitle('GEOS Carbon Reanalysis\n%s - %s' % (start_date, end_date), fontsize=14, fontweight='bold', y=1.02)
        lbl = fig.text(0.5,-0.01, 'Lat = %.2f, Lon = %.2f, Location = %s' % (float(lat), float(lon), station), ha='center', fontsize=11, fontweight='bold')

        # logos
        ax = fig.add_axes([0, 1, 1, 0.05], anchor='NE', zorder=-1)
        ax.axis('off')

        img = '_'.join(['carbon',product, str(lat), str(lon)])+'.png'
        plt.savefig(os.path.join(opath, img), bbox_inches='tight', dpi=100) #, bbox_extra_artists=(lbl,te_,te_1,te))
        print('Saving...'+img)

        plt.close()

        # logos
        gmao_logo = '/discover/nobackup/dao_ops/ebsmith2/gram/GMAO-logo_small.png'
        nasa_logo = '/discover/nobackup/dao_ops/ebsmith2/gram/nasa-logo_small.png'
        os.system('composite -gravity northeast -geometry +35+30 ' + gmao_logo + ' ' + os.path.join(opath, img) + ' ' + os.path.join(opath, img))
        os.system('composite -gravity northwest -geometry +40+15 ' + nasa_logo + ' ' + os.path.join(opath, img) + ' ' + os.path.join(opath, img))

    del cldtot
    #del cldhgh, cldmid, cldlow, cldhgh_t
    #del u, v, slp, t2m
    #del rh, ocext, bcext, suext, co, co2, airdens, ssext, duext
    #del ssexttau, duexttau, bcexttau, ocexttau, suexttau
    #del sss, dus, bcs, ocs, so4
    #del ssmass, dumass, ocmass, bcmass, sumass, nimass
    #del prectot, precsno, preccon, u2m, v2m, precip_t

def get_data(ipath, field, station, collection=None):
    try:
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
