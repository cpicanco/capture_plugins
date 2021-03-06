# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

# hacked from Screen_Marker_Calibration

import os
import cv2
import numpy as np
from methods import normalize,denormalize
from gl_utils import adjust_gl_view,clear_gl_screen,basic_gl_setup
import OpenGL.GL as gl
from glfw import *
from circle_detector import CircleTracker
from file_methods import load_object,save_object
from platform import system
from random import shuffle

from pyglui import ui
from pyglui.cygl.utils import draw_points, draw_polyline, RGBA
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path
from calibration_routines.calibration_plugin_base import Calibration_Plugin
from calibration_routines.finish_calibration import finish_calibration
from calibration_routines.screen_marker_calibration import interp_fn

#logging
import logging
logger = logging.getLogger(__name__)

# window calbacks
def on_resize(window,w,h):
    active_window = glfwGetCurrentContext()
    glfwMakeContextCurrent(window)
    adjust_gl_view(w,h)
    glfwMakeContextCurrent(active_window)

class Participant_Driven_Screen_Marker_Calibration(Calibration_Plugin):
    """
    Calibrate using on screen markers. 
    We use a ring detector that moves across the screen to 9 sites
    Points are collected at sites - not between
    Points are collected after space key is pressed

    """
    def __init__(
            self, g_pool,
            fullscreen=True,
            marker_scale=1.0,
            sample_duration=40,
            monitor_idx=1
        ):
        super().__init__(g_pool)
        self.detected = False
        self.space_key_was_pressed = False
        self.screen_marker_state = 0.
        self.sample_duration =  sample_duration # number of frames to sample per site
        self.fixation_boost = sample_duration/2.
        self.lead_in = 25 #frames of marker shown before starting to sample
        self.lead_out = 5 #frames of markers shown after sampling is donw
        self.monitor_idx = monitor_idx

        self.active_site = None
        self.sites = []
        self.display_pos = -1., -1.
        self.on_position = False
        self.pos = None
        self.marker_scale = marker_scale

        self._window = None
        self.menu = None
        self.button = None
        self.fullscreen = fullscreen
        self.clicks_to_close = 5

        self.glfont = fontstash.Context()
        self.glfont.add_font('opensans',get_opensans_font_path())
        self.glfont.set_size(32)
        self.glfont.set_color_float((0.2,0.5,0.9,1.0))
        self.glfont.set_align_string(v_align='center')

        # UI Platform tweaks
        if system() == 'Linux':
            self.window_position_default = (0, 0)
        elif system() == 'Windows':
            self.window_position_default = (8, 31)
        else:
            self.window_position_default = (0, 0)
        
        self.circle_tracker = CircleTracker()
        self.markers = []

    def init_ui(self):
        super().init_ui()
        self.menu.label = "Participant Driven Screen Marker Calibration"
        self.monitor_names = [glfwGetMonitorName(m) for m in glfwGetMonitors()]
        self.menu.append(ui.Info_Text("Calibrate gaze parameters pressing space key for each marker."))
        self.menu.append(ui.Selector('monitor_idx',self,selection = range(len(self.monitor_names)),labels=self.monitor_names,label='Monitor'))
        self.menu.append(ui.Switch('fullscreen',self,label='Use fullscreen'))
        self.menu.append(ui.Slider('marker_scale',self,step=0.1,min=0.5,max=2.0,label='Marker size'))
        self.menu.append(ui.Slider('sample_duration',self,step=1,min=10,max=100,label='Sample duration'))

    def start(self):
        if not self.g_pool.capture.online:
            logger.error("{} requires world capture video input.".format(self.mode_pretty))
            return
        super().start()
        logger.info("Starting {}".format(self.mode_pretty))

        if self.mode == 'calibration':
            self.sites = [(.5,  .5), (.25, .5), (0, .5), (1., .5), (.75, .5),
                          (.5, 1.), (.25, 1.), (0.,  1.),  (1., 1.), (.75, 1.),
                          (.25, .0), (1.,  0.), (.5, 0.), (0., 0.),  (.75, .0)
                          ]
            shuffle(self.sites)

        else:
            self.sites = [(.5, .5), (.25, .25), (.25, .75), (.75, .75), (.75, .25)]

        self.active_site = self.sites.pop(0)
        self.active = True
        self.ref_list = []
        self.pupil_list = []
        self.clicks_to_close = 5
        self.open_window(self.mode_pretty)

    def open_window(self,title='new_window'):
        if not self._window:
            if self.fullscreen:
                monitor = glfwGetMonitors()[self.monitor_idx]
                width, height, redBits, blueBits, greenBits, refreshRate = glfwGetVideoMode(monitor)
            else:
                monitor = None
                width,height= 640,360

            self._window = glfwCreateWindow(width, height, title, monitor=monitor, share=glfwGetCurrentContext())
            if not self.fullscreen:
                glfwSetWindowPos(self._window, self.window_position_default[0], self.window_position_default[1])

            glfwSetInputMode(self._window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN)

            # Register callbacks
            glfwSetFramebufferSizeCallback(self._window, on_resize)
            glfwSetKeyCallback(self._window, self.on_window_key)
            glfwSetMouseButtonCallback(self._window, self.on_window_mouse_button)
            on_resize(self._window, *glfwGetFramebufferSize(self._window))

            # gl_state settings
            active_window = glfwGetCurrentContext()
            glfwMakeContextCurrent(self._window)
            basic_gl_setup()
            # refresh speed settings
            glfwSwapInterval(0)

            glfwMakeContextCurrent(active_window)

    def on_window_key(self,window, key, scancode, action, mods):
        if action == GLFW_PRESS:
            if key == GLFW_KEY_ESCAPE:
                self.clicks_to_close = 0

            if key == GLFW_KEY_SPACE:
                self.space_key_was_pressed = True

    def on_window_mouse_button(self,window,button, action, mods):
        if action ==GLFW_PRESS:
            self.clicks_to_close -=1

    def stop(self):
        # TODO: redundancy between all gaze mappers -> might be moved to parent class
        logger.info("Stopping {}".format(self.mode_pretty))
        self.smooth_pos = 0, 0
        self.counter = 0
        self.close_window()
        self.active = False
        self.button.status_text = ''
        if self.mode == 'calibration':
            finish_calibration(self.g_pool, self.pupil_list, self.ref_list)
        elif self.mode == 'accuracy_test':
            self.finish_accuracy_test(self.pupil_list, self.ref_list)
        super().stop()

    def close_window(self):
        if self._window:
            # enable mouse display
            active_window = glfwGetCurrentContext();
            glfwSetInputMode(self._window,GLFW_CURSOR,GLFW_CURSOR_NORMAL)
            glfwDestroyWindow(self._window)
            self._window = None
            glfwMakeContextCurrent(active_window)

    def recent_events(self, events):
        frame = events.get('frame')
        if self.active and frame:
            recent_pupil_positions = events['pupil_positions']
            gray_img = frame.gray

            if self.clicks_to_close <=0:
                self.stop()
                return

            # Update the marker
            self.markers = self.circle_tracker.update(gray_img)
            # Screen marker takes only Ref marker
            self.markers = [marker for marker in self.markers if marker['marker_type'] == 'Ref']

            if len(self.markers) > 0:
                self.detected = True
                # Set the pos to be the center of the first detected marker
                marker_pos = self.markers[0]['img_pos']
                self.pos = self.markers[0]['norm_pos']
            else:
                self.detected = False
                self.pos = None  # indicate that no reference is detected

            if len(self.markers) > 1:
                logger.warning("{} markers detected. Please remove all the other markers".format(len(self.markers)))

            # only save a valid ref position if within sample window of calibraiton routine
            on_position = self.lead_in < self.screen_marker_state < (self.lead_in+self.sample_duration)
            
            if on_position and self.detected and self.space_key_was_pressed:
                ref = {}
                ref["norm_pos"] = self.pos
                ref["screen_pos"] = marker_pos
                ref["timestamp"] = frame.timestamp
                self.ref_list.append(ref)

            # always save pupil positions
            for p_pt in recent_pupil_positions:
                if p_pt['confidence'] > self.pupil_confidence_threshold:
                    self.pupil_list.append(p_pt)

            if on_position and self.detected and events.get('fixations', []) and self.space_key_was_pressed:
                self.screen_marker_state = min(
                    self.sample_duration+self.lead_in,
                    self.screen_marker_state+self.fixation_boost)

            # Animate the screen marker
            if self.screen_marker_state < self.sample_duration+self.lead_in+self.lead_out:
                if (self.detected and self.space_key_was_pressed) or not on_position:
                    self.screen_marker_state += 1
            else:
                self.space_key_was_pressed = False
                self.screen_marker_state = 0
                if not self.sites:
                    self.stop()
                    return
                self.active_site = self.sites.pop(0)
                logger.debug("Moving screen marker to site at {} {}".format(*self.active_site))

            # use np.arrays for per element wise math
            self.display_pos = np.array(self.active_site)
            self.on_position = on_position
            self.button.status_text = '{} / {}'.format(self.active_site, 15)

        if self._window:
            self.gl_display_in_window()

    def gl_display(self):
        """
        use gl calls to render
        at least:
            the published position of the reference
        better:
            show the detected postion even if not published
        """

        # debug mode within world will show green ellipses around detected ellipses
        if self.active and self.detected:
            for marker in self.markers:
                e = marker['ellipses'][-1] # outermost ellipse
                pts = cv2.ellipse2Poly((int(e[0][0]), int(e[0][1])),
                                       (int(e[1][0]/2), int(e[1][1]/2)),
                                        int(e[-1]), 0, 360, 15)
                draw_polyline(pts, 1, RGBA(0.,1.,0.,1.))
                if len(self.markers) > 1:
                   draw_polyline(pts, 1, RGBA(1., 0., 0., .5), line_type=gl.GL_POLYGON)

    def gl_display_in_window(self):
        active_window = glfwGetCurrentContext()
        if glfwWindowShouldClose(self._window):
            self.close_window()
            return

        glfwMakeContextCurrent(self._window)

        clear_gl_screen()

        hdpi_factor = glfwGetFramebufferSize(self._window)[0]/glfwGetWindowSize(self._window)[0]
        r = self.marker_scale * hdpi_factor
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        p_window_size = glfwGetFramebufferSize(self._window)
        gl.glOrtho(0, p_window_size[0], p_window_size[1], 0, -1, 1)
        # Switch back to Model View Matrix
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        def map_value(value,in_range=(0,1),out_range=(0,1)):
            ratio = (out_range[1]-out_range[0])/(in_range[1]-in_range[0])
            return (value-in_range[0])*ratio+out_range[0]

        pad = 90 * r
        screen_pos = map_value(self.display_pos[0],out_range=(pad,p_window_size[0]-pad)),map_value(self.display_pos[1],out_range=(p_window_size[1]-pad,pad))
        alpha = interp_fn(self.screen_marker_state,0.,1.,float(self.sample_duration+self.lead_in+self.lead_out),float(self.lead_in),float(self.sample_duration+self.lead_in))

        r2 = 2 * r
        draw_points([screen_pos], size=60*r2, color=RGBA(0., 0., 0., alpha), sharpness=0.9)
        draw_points([screen_pos], size=38*r2, color=RGBA(1., 1., 1., alpha), sharpness=0.8)
        draw_points([screen_pos], size=19*r2, color=RGBA(0., 0., 0., alpha), sharpness=0.55)

        # some feedback on the detection state and button pressing
        if self.detected and self.on_position and self.space_key_was_pressed:
            color = RGBA(.8,.8,0., alpha)
        else:
            if self.detected:
                color = RGBA(0.,.8,0., alpha)
            else:
                color = RGBA(.8,0.,0., alpha)
        draw_points([screen_pos],size=3*r2,color=color,sharpness=0.5)

        if self.clicks_to_close <5:
            self.glfont.set_size(int(p_window_size[0]/30.))
            self.glfont.draw_text(p_window_size[0]/2.,p_window_size[1]/4.,'Touch {} more times to cancel {}.'.format(self.clicks_to_close, self.mode_pretty))

        glfwSwapBuffers(self._window)
        glfwMakeContextCurrent(active_window)

    def get_init_dict(self):
        d = {}
        d['fullscreen'] = self.fullscreen
        d['marker_scale'] = self.marker_scale
        d['sample_duration'] = self.sample_duration
        d['monitor_idx'] = self.monitor_idx
        return d

    def deinit_ui(self):
        """gets called when the plugin get terminated.
           either voluntarily or forced.
        """
        if self.active:
            self.stop()
        if self._window:
            self.close_window()
        super().deinit_ui()

del Calibration_Plugin