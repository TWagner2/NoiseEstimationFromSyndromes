#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:48:13 2020

@author: thomas
"""

import multiprocessing as mp
import time
import sys
#see: https://pymotw.com/2/multiprocessing/communication.html
class QueueWorker(mp.Process):
    def __init__(self,TaskQueue,ResultQueue,Print = False):
        mp.Process.__init__(self)
        self.TaskQueue = TaskQueue
        self.ResultQueue = ResultQueue
        self.Print = Print
    def run(self):
        print("Starting Worker: ", self.name)
        sys.stdout.flush()
        while True:
            NextTask = self.TaskQueue.get()
            if NextTask is None:
                self.TaskQueue.task_done()
                break
            result = NextTask()
            self.TaskQueue.task_done()
            if self.Print:
                print("Tasks remaining: {}".format(self.TaskQueue.qsize()))
                sys.stdout.flush()
            self.ResultQueue.put(result)
        print("Exiting: ", self.name)
        sys.stdout.flush()
        return