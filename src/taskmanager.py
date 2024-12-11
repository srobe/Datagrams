from __future__ import print_function
# should move print statements outside of module

import time
import shlex
import subprocess
import multiprocessing

class TaskManager(object):
    def __init__(self): # bad default
        self.max_task = multiprocessing.cpu_count()
        print('Using ' + str(self.max_task) + ' processors.')
        self.task = {}
        self.command = {}

    def spawn(self, command):
        while self.load() >= self.max_task:
            time.sleep(0.001)

        cmd = shlex.split(command)

        p = subprocess.Popen(
            cmd, stdout=None, stderr=None, shell=False
        )
        process = str(p.pid)
        self.task[process] = p
        self.command[process] = command

    def load(self):
        pdelete = []
        ntask = len(self.task)
        processes = self.task.keys()

        for process in processes:
            p = self.task[process]

            if p.poll() == None:
                continue

            if p.returncode != 0:
                print(str(self.command[process]) + 'ended with rc=' + str(p.returncode))
            pdelete.append(process)

            ntask -= 1

        for process in pdelete:
            del self.task[process]
            del self.command[process]

        return ntask

    def wait(self):
        while self.load() > 0:
            exit_codes = [self.task[p].wait() for p in self.task.keys()]
