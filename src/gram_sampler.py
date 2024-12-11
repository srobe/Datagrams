#! /usr/bin/env python

import os
import re
import sys
import glob
import datetime as dt
import argparse

import config
import myutils
import tempdisk
import gradstime
import taskmanager
import stationmanager

# Retrieve command-line arguments
# ===============================

parser = argparse.ArgumentParser(description='Creates station profiles from gridded GEOS data')
parser.add_argument('datetime', metavar='datetime', type=str,
                    help='ISO datetime as ccyy-mm-ddThh:mm:ss')
parser.add_argument('config', metavar='config', type=str,
                    help='configuration file (.yml)')
parser.add_argument('--tau', metavar='tau', type=int, required=False,
                    help='hours', default=0) 

args = parser.parse_args()

dattim = re.sub('[^0-9]','', args.datetime+'000000')
idate  = int(dattim[0:8])
itime  = int(dattim[8:14])
tau    = args.tau

# Get configuration.
# ==================

cfg = config.Config()
cfg.read(args.config)

# Get environment definitions
# ===========================

ut = myutils.Utils()
gt = gradstime.GradsTime(idate,itime)

defs = { k:str(v) for k,v in iter(os.environ.items()) }
defs.update( {k:str(v) for k,v in iter(cfg.items()) if not isinstance(v,dict)} )
defs.update(cfg.get('environment',{}))

# Get station names and locations.
# Combine into one file.
# ================================

tmp   = tempdisk.TempDisk()
sta   = stationmanager.Station()
spath = cfg['STATION_PATH']

stations = sta.find(spath)
sfile    = tmp.get_file('stations')
sta.write(sfile, stations)

# Interpolate forecast data to the
# station locations and save.
# ================================

collections = cfg['collections']
flen = cfg.get('FC_LENGTH', 5)
task = taskmanager.TaskManager()

for cname,collection in iter(collections.items()):

    defs['collection'] = cname

    cfile  = collection.get('file', '$FC_GRID_PATHNAME')
    vars   = collection.get('vars', [])
    levs   = str(collection.get('levs', '')).split('-')
    offset = collection.get('offset', 0)
    stime  = gt.strvtime("%y4%m2%d2T%h2%n2%s2", 0, offset)
    etime  = gt.strvtime("%y4%m2%d2T%h2%n2%s2", flen*24, -offset)

    if levs[0]:
        levs = '-s ' + levs[0] + ' -e ' + levs[-1]
    else:
        levs = ''

    for var in vars:

        method = collection.get('method', 'nearest')

        if isinstance(var, dict):
            name   = list(var)[0]
            method = var[name].get('method', method)
            cfile  = var[name].get('file', cfile)
            var    = name

        defs['var'] = var
        cfile  = gt.strftime(ut.replace(cfile, **defs), tau)
        ofile  = gt.strftime(ut.replace('$FC_PROF_PATHNAME', **defs), tau)

        odir = os.path.dirname(ofile)
        try:
            os.makedirs(odir, 0o755)
        except:
            pass

        command = ['stn_sampler.csh', sfile, cfile, stime, etime, '-V', var, '-a', method, '-v', '-o', ofile, levs]
        command = ' '.join(command)

        task.spawn(command)
        print(command)

task.wait()
