import time
import types
import functools
import builtins
import ctypes
from multiprocessing import Process, Queue, Value
from collections import deque 
from pdb import set_trace

import pygame
import numpy as np

class ExplicitFeedbackListener(Process):
    def __init__(self, fps=10, video_size=[200, 100]):
        super(Process, self).__init__()
        self.video_size = video_size
        self.queue = Queue()
        self.fps = fps
        self.listening = Value(ctypes.c_bool, False)
         
    def run(self):
        self._init_pygames()
        self.listening.value = True

        while self.listening.value:
            fb, fill = self._do_pygame_events()
            if fb != 0:
                self.queue.put(dict(feedback = fb, time = time.time()))
                
            self._update_screen(fill)
            self.clock.tick(self.fps)
        self.close()
        
    def _init_pygames(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.video_size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self._update_screen()
    
    def _do_pygame_events(self):
        fb, fill = 0, None
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    fill = self.screen.fill((0, 255, 0))
                    fb = 1
                elif event.key == pygame.K_2:
                    fill = self.screen.fill((255, 0, 0))
                    fb = -1
            elif event.type == pygame.VIDEORESIZE:
                self.video_size = event.size
                self._update_screen(fill)
            elif event.type == pygame.QUIT:
                self.listening.value = False
        return fb, fill 
    
    def _update_screen(self, fill=None):
        if fill is None:
            fill = self.screen.fill((0, 0, 0))
        pygame.display.update(fill) 
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

        