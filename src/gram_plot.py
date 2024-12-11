#! /usr/bin/env python

import os
import re
import sys
import glob
import argparse
import importlib
import datetime as dt

import config
import myutils
import gradstime
import stationmanager

# Retrieve command-line arguments
# ===============================

parser = argparse.ArgumentParser(description='Creates Datagram Images')
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
# ================================

sta      = stationmanager.Station()
spath    = cfg['STATION_PATH']
stations = sta.find(spath)

# Create data grams
# =================

in_path  = gt.strftime(ut.replace('$FC_PROF_PATH', **defs), tau)
out_path = gt.strftime(ut.replace('$GRAM_IMG_PATH', **defs), tau)

try:
    os.makedirs(out_path, 0o755)
except:
    pass

plot = importlib.import_module(cfg.get('PLOT_ENGINE','plot_fp'))
plot.do_plots(in_path, stations, gt.idt, out_path)
