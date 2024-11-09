from typing import Dict
from pdb import set_trace

import sys
import gymnasium
sys.modules['gym'] = gymnasium

import gymnasium as gym
import numpy as np
import pygame
import pygame_gui
from imitation.data.rollout import VecEnv

class PyGameEventHandler():
    event_callbacks = []

    @classmethod
    def process_events(cls):
        for event in pygame.event.get():
            [cb(event) for cb in cls.event_callbacks]

class PyGameRender(gym.Wrapper):
    """ Wrapper environment for force rending Gymnasium games using PyGames. 
    
        This wrapper bypasses method render() by rendering even when step() 
        is called. Optionally, can render a menu between episodes to store 
        which episodes should be dropped or not. Useful when paired with 
        PlayableRolloutCollector which can use said info to drop marked
        episodes. 
        
        Attributes:
            env: Gym environment wrapped using SB3 vector environment
            
            enable_menu: Enables menu for marking episodes to drop.
            
            screen_width: Starting width of screen to be rendered.
            
            screen_height: Starting height of screen to be rendered.
            
            transpose: Whether or not to transpose rendered image. Can be 
                needed for certain environments.
                
            fps: Target frames rate to maintain.

            video_size: Dimensions of screen to be rendered.
             
    """
    # TODO: Automatically detect these objects by searching for pygame.Surface and
    #       pygame.time.Clock when closing.
    _base_env_screen_names = ['screen', 'clock', 'surf']
    def __init__(
        self, 
        env: VecEnv, 
        enable_menu: bool = False,
        screen_width: int = 600, 
        screen_height: int = 400, 
        transpose: bool = True, 
        fps: int = None,
    ):
        super().__init__(env)
        self.enable_menu = enable_menu
        self._episode = 0
        self._n_episodes = 0
        self._step = 0
        self.screen_width = screen_width 
        self.screen_height = screen_height
        self.screen = None
        self.clock = None
        self.transpose = transpose
        self.fps = int(fps) if fps is not None else self.metadata.get('render_fps', 30) 
        self.video_size = (int(self.screen_width), int(self.screen_height))
        self.keep_episodes = []
        self.show = False
        self.waiting = False
 
        if self.render_mode != 'rgb_array':
            gym.logger.warn(
                f"You using the PyGameRender wrapper with render mode {self.render_mode}. "
                "You can only specify the render_mode as 'rgb_array', "
                f'e.g., gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            
    def is_pygame_initialized(self):
        """ Check if pygame.init() has already been called. """
        try:
            pygame.display.get_init()
            return True
        except pygame.error:
            return False
    
    def _init_pygame(self) -> None:
        """ Initialize PyGame settings """
        if self.screen is None:
            # if self.is_pygame_initialized():
            #     msg = "PyGame has already been initialized. Make sure the base " \
            #           "environment is not doing so."
            #     raise pygame.error(msg)
            pygame.init()
            self._init_screen()
            if self.enable_menu: 
                self._init_menu_interface()
                PyGameEventHandler.event_callbacks.append(
                    self._check_menu_interface_event
                )
            PyGameEventHandler.event_callbacks.extend(
                [self._process_events, self.manager.process_events]
            )
            # self.isopen = True
            
        if self.clock is None:
            self.clock = pygame.time.Clock()
            # if hasattr(self.env.unwrapped, 'clock'):
            #     self.env.unwrapped.clock = self.clock
            
    def _init_screen(self) -> None:
        """ Method to run once to initialize PyGame screen or surfaces
        
            Override this method to add new screen or surface initializations.
        """
        self.screen = pygame.display.set_mode(
            self.video_size, pygame.RESIZABLE
        )
        # if hasattr(self.env.unwrapped, 'screen'):
        #     self.env.unwrapped.screen = self.screen
        self.manager = pygame_gui.UIManager(self.video_size)
    
    def _init_menu_interface(self) -> None:
        """ Initializes the menu interface for continuing or replaying
        
            Here the 'continue' button will move onto the next trajectory while 
            the 'replay' button will replay the current trajectory.
            
            Additional buttons can be added and initialized by overriding this method.
        """
        # Initialize replay interfaced
        self.replay_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(0, 0, 100, 30),
            text='Replay',
            manager=self.manager,
            visible=False,
            anchors={
                'center': 'center',
            }
        )
        self.continue_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(0, -30, 100, 30),
            text='Continue',
            manager=self.manager,
            visible=False,
            anchors={
                'center': 'center',
                'bottom': 'bottom',
                'bottom_target': self.replay_button
            }
        )
    
    def reset(self, **kwargs):
        """ Reset Gymnasium env and run menu interface between episodes """
        self._init_pygame()
        if self._episode != 0:
            if self.enable_menu:
                # Use menu to decide if trajectory is kept or not.
                self._run_menu_interface(self.env.render())
            else:
                # Use every trajectory if no menu.
                self.keep_episodes.append(self._n_episodes-1)
        # Calls any wrappers BELOW this wrapper
        output = self.env.reset(**kwargs)
        # Update privated info
        self._episode += 1 # Tracks current episode
        self._n_episodes += 1 # Tracks total episodes
        self._step = 0
        # Render reset screen
        self.force_render()
        
        return output
    
    def _run_menu_interface(self, rendered = None) -> None:
        """ Logic for running the menu interface """
        self._show_menu_interface(True)
        self._wait_for_replay_response()
        self._show_menu_interface(False)
    
    def _show_menu_interface(self, show: bool) -> None:
        """ Shows and hides the menu interface buttons 
        
            Args:
                show: If true, the menu interface buttons will be shown, otherwise
                    passing false hides the buttons.
        """
        self.show = show
        if show:
            self.replay_button.show()
            self.continue_button.show()
        else:
            self.replay_button.hide()
            self.continue_button.hide()
    
    def _wait_for_replay_response(self, rendered= None) -> None:
        """ Loops the menu interface until a response is given """
        
        self.waiting = True
        while self.waiting:
            self.time_delta = self._tick()
            # self._process_menu_interface_events()
            PyGameEventHandler.process_events()
            self.force_render()
        
    # def _process_menu_interface_events(self) -> bool:
    #     """ Process events specific to the menu interface 
        
    #         Returns:
    #             Whether or not waiting should continue. If false, waiting will end.
    #             If true, waiting will continue.
    #     """
    #     # Events needed to detect button push for pygame GUI
    #     event_list = [
    #         pygame_gui.UI_BUTTON_PRESSED,
    #         pygame.MOUSEBUTTONDOWN, 
    #         pygame.MOUSEBUTTONUP,
    #     ]
    #     for event in pygame.event.get(event_list):
    #         self._check_menu_interface_event(event)
            # self.manager.process_events(event)
    
    def _check_menu_interface_event(self, event: pygame.event.Event) -> bool:
        """ Default events specific to the menu interface.
        
            Args:
                Event: Pygame event
        
            Returns:
                Whether or not waiting should continue. If false, waiting will end.
                If true, waiting will continue.
        """
        if self.show and event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.replay_button:
                self._episode -= 1
                self.waiting = False
            if event.ui_element == self.continue_button:
                self.keep_episodes.append(self._n_episodes-1)
                self.waiting = False

    def step(self, action):
        self._tick()
        output = self.env.step(action)
        PyGameEventHandler.process_events()
        self.force_render()
        self._step += 1
        # assert self.time_delta == self.get_time()
        return output

    def force_render(self, rendered: np.ndarray = None):
        """Renders environment using environment render or passed image. 
        
            Args:
                rendered: RGB NumPy array for an image. 
        """
        rendered = self.env.render() if rendered is None else rendered
        # self._process_events()
        self.manager.update(self.get_time()) # GUI related
        self._draw_screen(self.screen, rendered, self.video_size, self.transpose)
        self.manager.draw_ui(window_surface=self.screen) # GUI related
        pygame.display.set_caption(self._get_caption())
        pygame.display.flip()
        
    def _tick(self) -> None:
        """ Delays PyGame frame to keep game running at desired FPS.
        
        
            Return:
                Milliseconds since the previous call to Clock.tick()
        """
        return self.clock.tick(self.fps)/1000.0
    
    def get_time(self) -> None:
        """ The number of milliseconds that passed between the previous two calls to Clock.tick().
        
            Return:
                Milliseconds since the previous call to Clock.tick()
        """
        return self.clock.get_time()/1000.0
    
    def _process_events(self, event):
        # for event in pygame.event.get([pygame.QUIT, pygame.VIDEORESIZE]):
        if event.type == pygame.QUIT:
            self.close()
        if event.type == pygame.VIDEORESIZE:
            self.video_size = (event.w, event.h)
            self.manager.set_window_resolution(self.video_size)
            
    def _draw_screen(self, screen, arr, video_size, transpose):
        """ Updates the current screen using image array
        
            Args:
                
        """
        arr_min, arr_max = np.min(arr), np.max(arr)
        if arr_max != 0 and arr_min != 0:
            arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)

        pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
        pyg_img = pygame.transform.scale(pyg_img, video_size)
        
        # We might have to add black bars if surface_size is larger than video_size
        surface_size = screen.get_size()
        width_offset = (surface_size[0] - video_size[0]) / 2
        height_offset = (surface_size[1] - video_size[1]) / 2
        
        screen.fill((0, 0, 0))
        screen.blit(pyg_img, (width_offset, height_offset))
    
    def _get_caption(self):
        """ Updates the caption located at the top of the PyGame window """
        caption = f"FPS:{round(self.clock.get_fps())}"
        caption += f" Episode:{self._episode}/{self._n_episodes}"
        caption += f" Step:{self._step}"
        return caption

    def close(self):
        """ Closes pygame safely

            This method should be ran in order to shutdown PyGame properly.
        """
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            [setattr(self.env.unwrapped, name, None) for name in self._base_env_screen_names
                if hasattr(self.env.unwrapped, name)]
            self.screen = None
            self.clock = None
            # self.isopen = False