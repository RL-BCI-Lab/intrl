 
import os
import re
import time
from abc import abstractmethod
from typing import Callable, List, Dict, Optional, Tuple
from os.path import join, isfile
from pathlib import Path
from pdb import set_trace

import pygame
import pygame_gui
import numpy as np
from imitation.data.rollout import flatten_trajectories
from imitation.data.types import Trajectory, Transitions

from intrl.common.imitation.utils import extract_keys_from_trajectory_info
from intrl.common.data.storage import H5Storage

class Replayer():
    _render_key = 'render'
    """ Base class for Replaying pre-captured trajectories. 
    
        Attributes:        
            trajectories: A list containing trajectory objects for holding Markov tuple.
    """
    def __init__(self, trajectories: List[Trajectory]) -> None:
        self.trajectories = trajectories
        self._render_validation(trajectories)

    def _render_validation(self, trajectories: List[Trajectory]) -> None:
        """ Validates that trajectories have correct info.
        
            First checks that each trajectory has the 'infos' key. Then checks if each 
            info key has the required `_render_key` which will hold the image for rendering. 
        """
        has_infos = np.all([hasattr(traj, 'infos') and traj.infos is not None for traj in trajectories])
        if not has_infos:
            raise AttributeError("Trajectories are missing key 'info'")
        
        has_render = np.all([self._render_key in info for traj in trajectories for info in traj.infos])
        if not has_render:
            raise KeyError(f"Trajectory info is missing key '{self._render_key}'")
    
    
    @abstractmethod
    def play(self):
        """ Replays passed trajectories.
        
            This method should contain the logic for playing the passed trajectories.
        """
        pass
    
    
    def close(self):
        """ Wraps up any loose ends once playing is down. 
        
            This should only be called once all trajectories have been played or this
            class is no longer needed so that any elements can be safely closed.
        """
        pass


class PyGameReplayer(Replayer):
    """ Replays trajectories using PyGame interface.
    
        Attributes:
            trajectories: A list containing trajectory objects for holding Markov tuple.
            
            transpose: Determines whether images should be transposed or not when being
                displayed.
            
            fps: Target frames-per-second (FPS) at which the images will be displayed.
            
            zoom: Zoom factor used to increase display size of images.
            
            clock: PyGame clock for tracking elasped time.
            
            screen: PyGame screen.
            
            manager: PyGame UI manager for managing additional UI elements.
            
            isopen: Boolean for determining if the close() method has been called.
            
            shutdown: Boolean for breaking out of the play() method loop.
            
            replay: Boolean for determining whether the current trajectory should be 
                replayed. This is determined via the UI.
    """
    def __init__(
        self, 
        trajectories: List[Trajectory], 
        transpose: bool = True, 
        fps: int = 30,
        zoom: float = 1,
    ) -> None:
        
        super().__init__(trajectories)
        self.transpose = transpose
        self.video_size = self._get_video_size(zoom)
        self.fps = fps 
        self.clock = None
        self.screen = None
        self.manager = None
        self.isopen = True
        self.shutdown = False
        self._replay = None
        self.show = False
        self.waiting = False

    def _get_video_size(self, zoom: float = None) -> Tuple[float]:
        """ Get the initial screen size using the passed images
        
            Args:
                zoom: How much to zoom in on the rendered images. Zoom will increase
                    the size of the starting screen relative to the dimensions of the
                    images stored in 'render'.
                    
            Returns:
                Returns the size of the PyGame screen.
        """
        video_size = self.trajectories[0].infos[0][self._render_key].shape[:-1]

        if self.transpose:
            # Transpose original image video_size
            video_size = (video_size[1], video_size[0])

        if zoom is not None:
            video_size = (int(video_size[0] * zoom), int(video_size[1] * zoom))

        return video_size
    
    def play(self) -> None:
        """ Start replaying the loaded trajectories """
        self._init_pygame()
        
        for t, trajectory in enumerate(self.trajectories):
            self._play_trajectory(t, trajectory)
            
            if self.shutdown:
                break

        self.close()  
    
    def _init_pygame(self) -> None:
        """ Initialize PyGame settings """
        if self.screen is None:
            pygame.init()
            self._init_screen()
            self._init_menu_interface()
            
        if self.clock is None:
            self.clock = pygame.time.Clock()
    
    def _init_screen(self) -> None:
        """ Method to run once to initialize PyGame screen or surfaces
        
            Override this method to add new screen or surface initializations.
        """
        self.screen = pygame.display.set_mode(
            self.video_size, pygame.RESIZABLE
        )
        self.manager = pygame_gui.UIManager(self.video_size)
    
    def _init_menu_interface(self) -> None:
        """ Initializes the menu interface for continuing or replaying
        
            Here the 'continue' button will move onto the next trajectory while 
            the 'replay' button will replay the current trajectory.
            
            Additional buttons can be added and initialized by overriding this method.
        """
        # Initialize replay interface
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

    def _play_trajectory(self, episode: int, trajectory: Trajectory) -> None:
        """ Plays a single trajectory

            This method contains the logic for running the replay and managing the
            menu interface once the replay has finished.
            
            Args:
                trajectory; Current trajectory to be replayed.
                
                episode: The current episode number (i.e., the trajectory number)
        """
        self._replay = True
        pygame.event.clear()
        transitions = flatten_trajectories([trajectory])
        # Loop again only if replay is selected
        while self._replay:
            self._run_replay(episode, transitions)
            self._run_menu_interface(transitions[-1])
            pygame.event.clear()
            if self.shutdown:
                break
       
    def _run_replay(self, episode: int, transitions: Transitions) -> None:
        """ Logic for displaying each transition
        
            Args:
                transitions: Set of Markov tuples or transitions to be replayed.
                
                episode: The current episode number (i.e., the trajectory number)
        """
        for step, trans in enumerate(transitions):
            self._tick()
            self.before_render_transition(trans, episode, step)
            self._process_events()
            self.manager.update(self.get_time())
            self._draw_screen(trans)
            self.manager.draw_ui(self.screen)
            caption = self._get_caption(
                episode, 
                len(self.trajectories), 
                step, 
                len(transitions)
            )
            # Render screen updates
            pygame.display.set_caption(caption)
            pygame.display.flip()
            self.after_render_transition(trans, episode, step)
            if self.shutdown:
                return
    
    def _tick(self) -> None:
        """ Sets the PyGame clock and gets returns current frame number.
        
        
            Return:
                The current frame number.
        """
        return self.clock.tick(self.fps)/1000.0
    
    def get_time(self) -> None:
        """ The number of milliseconds that passed between the previous two calls to Clock.tick().
        
            Return:
                Milliseconds since the previous call to Clock.tick()
        """
        return self.clock.get_time()/1000.0
    
    def before_render_transition(self, transition: Dict, episode: int, step: int) -> None:
        """ Run code given the current transition before rendering occurs. 
        
            Can be useful to implement when inheriting to prevent complete overwriting
            play() method and having code redundancy.
            
            Args:
                transition: A dictionary containing a Markov tuple.
                
                episode: The current episode.
                
                step: The current step in the current episode.
        """
        return None
    
    def after_render_transition(self, transition: Dict, episode: int, step: int) -> None:
        """ Run code given the current transition after rendering occurs. 
        
            Can be useful to implement when inheriting to prevent complete overwriting
            play() method and having code redundancy.
            
            Args:
                transition: A dictionary containing a Markov tuple.
                
                episode: The current episode.
                
                step: The current step in the current episode.
        """
        return None
           
    def _process_events(self) -> None:
        """ Default events that must be processed
        
            This method is typically called last as it will clear ALL other events
        """
        for event in pygame.event.get():
            # Stop manually and end replay early
            if event.type == pygame.QUIT:
                self.shutdown = True
            if event.type == pygame.VIDEORESIZE:
                self.video_size = (event.w, event.h)
                self.manager.set_window_resolution(self.video_size)
            self._check_menu_interface_event(event)
            self.manager.process_events(event)
            
            
    def _run_menu_interface(self, transition: Dict) -> None:
        """ Logic for running the menu interface
        
            Args:
                transition: A dictionary containing a Markov tuple which will be used as
                    the background for the menu items.
        """
        self._show_menu_interface(True)
        self._wait_for_replay_response(transition=transition)
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
    
    def _wait_for_replay_response(self, transition: Dict) -> None:
        """ Loops the menu interface until a response is given 
        
            Args:
                transition: A dictionary containing a Markov tuple which will be used as
                    the background for the menu items.
        """
        self.waiting = True
        while self.waiting:
            self._tick()
            # self._process_menu_interface_events()
            self._process_events()
            self.manager.update(self.get_time())
            self._draw_screen(transition)
            self.manager.draw_ui(window_surface=self.screen)
            pygame.display.flip()
            
            if self.shutdown:
                break
    
    # def _process_menu_interface_events(self) -> bool:
    #     """ Process events specific to the menu interface 
        
    #         Returns:
    #             Whether or not waiting should continue. If false, waiting will end.
    #             If true, waiting will continue.
    #     """
    #     event_list = [
    #         pygame_gui.UI_BUTTON_PRESSED,
    #         pygame.MOUSEBUTTONDOWN, 
    #         pygame.MOUSEBUTTONUP,
    #     ]
    #     for event in pygame.event.get(event_list):
    #         self._check_menu_interface_event(event)
    #         self.manager.process_events(event)
        
    def _check_menu_interface_event(self, event: pygame.event.Event) -> bool:
        """ Default events specific to the menu interface.
        
            Args:
                Event: Queue containing pygame.event.Event objects.
        
            Returns:
                Whether or not waiting should continue. If false, waiting will end.
                If true, waiting will continue.
        """
        if self.show and event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.replay_button:
                self._replay = True
                self.waiting = False
            if event.ui_element == self.continue_button:
                self._replay = False
                self.waiting = False
    
    def _draw_screen(self, transition: Dict):
        """ Updates the current screen using the render key stored in infos
        
            Args:
                transition: A dictionary containing a Markov tuple.
        """
        arr = transition['infos'][self._render_key]
        arr_min, arr_max = np.min(arr), np.max(arr)
        if arr_max != 0 and arr_min != 0:
            arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
      
        pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if self.transpose else arr)
        pyg_img = pygame.transform.scale(pyg_img, self.video_size)
        
        surface_size = self.screen.get_size()
        width_offset = (surface_size[0] - self.video_size[0]) / 2
        height_offset = (surface_size[1] - self.video_size[1]) / 2
        
        self.screen.fill((0, 0, 0))
        self.screen.blit(pyg_img, (width_offset, height_offset))

    def _get_caption(self, episode: int, total_episodes: int, step:int, total_steps: int):
        """ Updates the caption located at the top of the PyGame window """
        caption = f"FPS:{round(self.clock.get_fps())}"
        caption += f" Episode: {(episode+1)}/{total_episodes}"
        caption += f" Step: {(step+1)}/{total_steps}"
        return caption

    def close(self):
        """ Closes pygame safely

            This method should be ran in order to shutdown PyGame properly.
        """
        if hasattr(self, 'screen') and self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
            self.isopen = False
        
    def __del__(self):
        self.close()


# NOTE: Current limitation is that labels are mapped to states directly. Capturing labels
#       and timesteps would be more robust but is extremely difficult to do since the replay
#       time and original state times will differ. Additionally, mismatches between the original
#       FPS and replay FPS can exaggerate this problem. Mapping directly to states, without
#       using the 'additive' mode for the labeler, means some feedback will be lost but feedback
#       opposite feedback can cancel one another out.
class LabelDemonstrations(PyGameReplayer):
    """Replays demonstrations and allows them to be labeled with feedback."""
    def __init__(
        self, 
        labeler: Callable,
        trajectories: List[Trajectory],
        *,
        label_color_map: Dict[str,str] = None,
        label_key: str = 'feedbacks',
        **kwargs
    ):
        """ 
            Args:
                labeler: A callable object that returns a label for a given transition. The
                        function must taken in 3 arguments: transition, episode, and step. It 
                        also must return a single label of any type.
                    
                trajectories: An list of trajectories that will be replayed and labeled.
                
                label_key: The name of the key which will contain the label and be stored
                    within the current transition's info.
                    
                label_color_map: A mapping of label value to color which will be displayed
                    on screen as visual feedback to the user to indicate which label they
                    provided.
                    
                kwargs: Kwargs corresponding to PyGameReplayer init method.
        """
        super().__init__(trajectories, **kwargs)
        self.labeler = labeler
        self.label_color_map = {} if label_color_map is None else label_color_map
        self.label_key = label_key
        self.clear = False
        self.current_label = None
        self._label_color = (0,0,0,0)
        self._fade_rate = 10
        self._circle_radius = 25
        self._circle_center = (30, 30)
        
        self.avg_episode_fps = [] # Average FPS for each episode (not including menu time)
        self._fps_total = 0 
        self._fps_calls = 0
        self._curr_episode = None

    @property
    def average_fps(self):
        if self._fps_calls != 0:
            return self._fps_total / self._fps_calls
        else:
            return 0
        
    def play(
        self,
        save_paths: str = None,
        extract_info_keys: List[str] = None,
    ) -> None:
        super().play()
        if save_paths:
            self.save(save_paths, extract_info_keys)
    
    def _init_menu_interface(self) -> None:
        super()._init_menu_interface()
        # Add Clear button to menu interface
        self.clear_and_replay = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(0, 30, 100, 30),
            text='Clear',
            manager=self.manager,
            visible=False,
            anchors={
                'center': 'center',
                'bottom': 'bottom',
                'bottom_target': self.replay_button
            }
        )
        
    def _run_menu_interface(self, transition: Dict):
        # Make sure clear defaults to False
        self.clear = False
        super()._run_menu_interface(transition)
    
    def _check_menu_interface_event(self, event: pygame.event.Event):
        super()._check_menu_interface_event(event)
        if self.show and event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.clear_and_replay:
                self._replay = True
                self.clear = True
                self.waiting = False
    
    def _show_menu_interface(self, show: bool):
        super()._show_menu_interface(show)
        if show:
            self.clear_and_replay.show()
        else:
            self.clear_and_replay.hide()
    
    def _draw_screen(self, transition: Dict):
        super()._draw_screen(transition)
        if len(self.label_color_map) > 0:
            self._draw_label(self.current_label)
            self.screen.blit(self.shape_surf, self.target_rect)
            
    def _init_screen(self):
        super()._init_screen()
        # Initialize label feedback surface for showing color of corresponding label received
        self.target_rect = pygame.Rect(
            self._circle_center, (self._circle_radius * 2, self._circle_radius * 2)
        )
        self.shape_surf = pygame.Surface(self.target_rect.size, pygame.SRCALPHA)
        
    def _draw_label(self, label: int):
        """ Displays circle according to the provided label's color 
        
            The colored circle is displayed in response to the most recent label feedback
            which was provided. The circle will fade at a linear rate once drawn.
            
        """
        label_color = self.label_color_map.get(label, None)
        
        if label_color is not None:
            self._label_color = [*label_color, 255]
            pygame.draw.circle(
                surface=self.shape_surf, 
                color=self._label_color,
                center=[self._circle_radius]*2,
                radius=self._circle_radius
            )
        else:
            alpha = self._label_color[-1] 
            # Once faded all the way, stop drawing
            if alpha == 0:
                return
            # Draw current faded color until alpha value is 0
            dt = self.clock.get_time()/1000
            dt_fade_rate = (255 - self._fade_rate) * dt
            self._label_color[-1]  = int(max(alpha - dt_fade_rate, 0))
            pygame.draw.circle(
                surface=self.shape_surf, 
                color=self._label_color,
                center=[self._circle_radius]*2,
                radius=self._circle_radius
            )
    
    def _play_trajectory(self, episode: int, trajectory: Trajectory) -> None:
        """ Plays a single trajectory (i.e., episode)

            This method contains the logic for running the replay and managing the
            menu interface once the replay has finished.
            
            Args:
                trajectory; Current trajectory to be replayed.
                
                episode: The current episode number (i.e., the trajectory number)
        """
        self._curr_episode = episode
        super()._play_trajectory(episode, trajectory)
        self.avg_episode_fps.append(self.average_fps)
        self._fps_calls = 0
        self._fps_total = 0
    
    def before_render_transition(self, transition: Dict, episode: int, step: int):
        """ Run code given the current transition before rendering occurs. 
        
            Can be useful to implement when inheriting to prevent completely overwriting
            play() method and having code redundancy.
            
            Args:
                transition: A dictionary containing the output from gym's step() method.
                
                episode: The current episode.
                
                step: The current step in the current episode.
        """
        infos = transition['infos']
        label = self.labeler(transition, episode, step)
        # As infos is a dictionary, changes are made in-place. If existing label is 
        # given within infos, only overwrite if label is not the default value.
        if not self.clear and label == self.labeler.default_label:
            if self.label_key not in infos:
                infos[self.label_key] = label
        else:
            infos[self.label_key] = label
        self.current_label = label
        
        self._track_episode_average_fps(episode=episode)
    
    def after_render_transition(self, transition: Dict, episode: int, step: int):
        """ Track the time at which the state is rendered """
        infos = transition['infos']
        infos['time'] = time.time()
    
    def _track_episode_average_fps(self, episode):
        if self._curr_episode is None:
            self._curr_episode = episode

        # Ignore 0s as FPS will be reported as 0 until the second call to clock.tick
        if self.clock.get_fps() != 0:
            self._fps_total += self.clock.get_fps()
            self._fps_calls += 1
            # print(np.round(self.average_fps, 2), self.clock.get_fps())
    
    def save(
        self,
        paths: str = None,
        extract_info_keys: List[str] = None,
    ):
        """ Save feedback labeled trajectories
        
            Args:
                paths: Path to save .npz file to.
                    
                extract_info_keys: Additional keys to
        """
        assert len(paths) == len(self.trajectories)
        for i, path in enumerate(paths):
            data = extract_keys_from_trajectory_info(
                trajectory=self.trajectories[i],
                extract_keys={
                    self.label_key:self.label_key, 
                    'time':'time'
                 },
            )
            group = H5Storage.find_group_name(self.trajectories[i], path=path)
            H5Storage.append_to_file(
                data={group: {
                        self.label_key: {
                            'feedbacks': data[self.label_key], 
                            'time': data['time'],
                            'attrs': {
                                f'avg_fps': self.avg_episode_fps[i],
                                f'target_fps': self.fps
                            }
                        },
                    }
                },
                path=path
            )

    
class PyGameLabeler():
    _base_key_to_label = {
        pygame.K_UP: 1,
        pygame.K_DOWN: -1
    }
    """ Label replay states using PyGame API 
    
        Default keys for labeling are the up arrow for label +1 and the down arrow 
        for the label -1.
    """
    def __init__(
        self, 
        key_to_label: dict = None, 
        default_label: int = 0, 
        additive: bool = True
    ):
        self.k2l = self._base_key_to_label if key_to_label is None else key_to_label
        self.default_label = default_label
        self.additive = additive
        self.keys = [*self.k2l.keys()]
        self.label_locs = dict()
        
    def __call__(self, transition: Dict, episode: int, step: int):
        """ Returns label based on key press 
        
            If you push multiple keys in one frame, the last key is used. If
            self.additive is True then the labels will be added together. This is rare
            if not impossible when FPS for replay is considerably high (30+).
        """
        if episode not in self.label_locs:
            self.label_locs[episode] = []
        
        label = self.default_label
        for event in pygame.event.get([pygame.KEYDOWN]):
            if event.key in self.k2l:
                if self.additive:
                    label += self.k2l[event.key]
                else:
                    label = self.k2l[event.key]
                self.label_locs[episode].append(step)
        return label