import os
import fnmatch

def find(path=os.getcwd(), ext='.txt'):
    '''Recursive search function top-down.'''
    for (root, dirs, files) in os.walk(path):
        for f in fnmatch.filter(files, '*'+ext):
            yield os.path.join(root, f)
