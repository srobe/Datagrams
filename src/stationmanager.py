import re
import os
import copy
import find
import sys

class Station(object):
    def __init__(self):
        self.name = []
        self.lon = []
        self.lat = []
        self.data = {}

    def find(self, path):

        if os.path.isfile(path):
            files = [path]
        else:
            # what if they have .csv extensions?
            files = find.find(path=path, ext='.csv') #'.txt')

        # why not use csv package?
        for f in files:
            with open(f) as fi:
                lines = fi.readlines()

            for line in lines:
                text = line.rstrip()
                comment = re.match(r'^#', text)
                blank = re.match(r'^ *$', text)
                header = re.match(r'^name,lon,lat$', text)
                definition = re.match(r'(.+),([-\d\.]+),([-\d\.]+)', text)

                if comment or blank or header:
                    continue
                elif definition:
                    name = definition.group(1)
                    lon = float(definition.group(2))
                    lat = float(definition.group(3))

                    self.name.append(name)
                    self.lon.append(lon)
                    self.lat.append(lat)

                    if name in self.data:
                        lon_d = self.data[name]['lon']
                        lat_d = self.data[name]['lat']

                        if (lon_d == lon) and (lat_d == lat):
                            continue
                    else:
                        self.data[name] = {'lon':lon, 'lat':lat}
                else:
                    continue
        # why return a copy?
        return self.data
        return copy.deepcopy(self.data)

    def write(self, file, stations):
        with open(file, 'w') as f:
            f.write('name,lon,lat\n')

            for name in stations:
                lon = str(stations[name]['lon'])
                lat = str(stations[name]['lat'])
                f.write(','.join([name,lon,lat]) + '\n')

class Error(Exception):
    '''Base calss for exceptions in this module.'''
    pass

class IOError(Error):
    def __init__(self, msg):
        self.msg = msg

class NamespaceError(Error):
    def __init__(self, name, lon1, lon2, lat1, lat2):
        coords = ','.join([str(lon1), str(lon2), str(lat1), str(lat2)])
        self.msg = 'Namespace Error: ' + name + ': ' + coords
