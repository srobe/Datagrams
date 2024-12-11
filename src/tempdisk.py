from __future__ import print_function
# should move print statements outside of module

import os
import tempfile

class TempDisk(object):
    def __init__(self, root=None):
        self.files = []
        # ignored pid directory since this is dynamic
        # ignored self.root as mkdtemp returns abspath
        self.tmpdir = tempfile.mkdtemp()
        print('Created temporary directory: '+ self.tmpdir)

    def get_file(self, name):
        file = os.path.join(self.tmpdir, name)
        self.files.append(file)
        return file

    def __str__(self):
        return '\n'.join(self.files)

    def __del__(self):
        #return
        print('\nCleaning up temporary storage...')
        for f in self.files:
            print ('deleting ' + f)
            if os.path.isfile(f):
                os.remove(f)

        print('deleting ' + self.tmpdir)
        os.rmdir(self.tmpdir)
