# -*- coding: utf-8 -*-
'''
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2016 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

# Screen_Marker_Calibration hack

import numpy as np
from methods import normalize
from gl_utils import clear_gl_screen,basic_gl_setup
import OpenGL.GL as gl
from glfw import *
from circle_detector import find_concetric_circles

from pyglui import ui
from pyglui.cygl.utils import draw_points, RGBA,draw_concentric_circles
from pyglui.ui import get_opensans_font_path

from calibration_routines.calibration_plugin_base import Calibration_Plugin
from calibration_routines.finish_calibration import finish_calibration
from calibration_routines.screen_marker_calibration import Screen_Marker_Calibration, interp_fn

class Participant_Driven_Screen_Marker_Calibration(Screen_Marker_Calibration):
    """
    Requires key press before each sampling

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.space_key_was_pressed = False

    def init_gui(self):
        self.monitor_idx = 0
        self.monitor_names = [glfwGetMonitorName(m) for m in glfwGetMonitors()]

        #primary_monitor = glfwGetPrimaryMonitor()
        self.info = ui.Info_Text("Calibrate gaze parameters using a screen based animation.")
        self.g_pool.calibration_menu.append(self.info)

        self.menu = ui.Growing_Menu('Controls')
        self.g_pool.calibration_menu.append(self.menu)
        self.menu.append(ui.Selector('monitor_idx',self,selection = range(len(self.monitor_names)),labels=self.monitor_names,label='Monitor'))
        self.menu.append(ui.Switch('fullscreen',self,label='Use fullscreen'))
        self.menu.append(ui.Slider('marker_scale',self,step=0.1,min=0.5,max=2.0,label='Marker size'))
        self.menu.append(ui.Slider('sample_duration',self,step=1,min=10,max=100,label='Sample duration'))

        self.button = ui.Thumb('active',self,label='C',setter=self.toggle,hotkey='c')
        self.button.on_color[:] = (.3,.2,1.,.9)
        self.g_pool.quickbar.insert(0,self.button)

    def on_key(self,window, key, scancode, action, mods):
        if action == GLFW_PRESS:
            if key == GLFW_KEY_ESCAPE:
                self.clicks_to_close = 0

            if key == GLFW_KEY_SPACE:
            	self.space_key_was_pressed = True

    def recent_events(self, events):
        frame = events.get('frame')
        if self.active and frame:
            recent_pupil_positions = events['pupil_positions']
            gray_img = frame.gray

            if self.clicks_to_close <=0:
                self.stop()
                return

            # detect the marker
            self.markers = find_concetric_circles(gray_img, min_ring_count=4)
            
            if len(self.markers) > 0:
                self.detected = True
                marker_pos = self.markers[0][0][0]  # first marker, innermost ellipse,center
                self.pos = normalize(marker_pos, (frame.width, frame.height), flip_y=True)

            else:
                self.detected = False
                self.pos = None  # indicate that no reference is detected

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
            self.button.status_text = '{} / {}'.format(self.active_site, 9)

    def gl_display_in_window(self):
        active_window = glfwGetCurrentContext()
        if glfwWindowShouldClose(self._window):
            self.close_window()
            return

        glfwMakeContextCurrent(self._window)

        clear_gl_screen()

        hdpi_factor = glfwGetFramebufferSize(self._window)[0]/glfwGetWindowSize(self._window)[0]
        r = 110*self.marker_scale * hdpi_factor
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        p_window_size = glfwGetWindowSize(self._window)
        gl.glOrtho(0,p_window_size[0],p_window_size[1],0 ,-1,1)
        # Switch back to Model View Matrix
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        def map_value(value,in_range=(0,1),out_range=(0,1)):
            ratio = (out_range[1]-out_range[0])/(in_range[1]-in_range[0])
            return (value-in_range[0])*ratio+out_range[0]

        pad = .7*r
        screen_pos = map_value(self.display_pos[0],out_range=(pad,p_window_size[0]-pad)),map_value(self.display_pos[1],out_range=(p_window_size[1]-pad,pad))
        alpha = interp_fn(self.screen_marker_state,0.,1.,float(self.sample_duration+self.lead_in+self.lead_out),float(self.lead_in),float(self.sample_duration+self.lead_in))

        draw_concentric_circles(screen_pos,r,4,alpha)
        #some feedback on the detection state

        if self.detected and self.on_position and self.space_key_was_pressed:
            draw_points([screen_pos],size=10*self.marker_scale,color=RGBA(.8,.8,0.,alpha),sharpness=0.5)
        else:
            if self.detected:
                draw_points([screen_pos],size=10*self.marker_scale,color=RGBA(0.,.8,0.,alpha),sharpness=0.5)
            else:
                draw_points([screen_pos],size=10*self.marker_scale,color=RGBA(.8,0.,0.,alpha),sharpness=0.5)

        if self.clicks_to_close <5:
            self.glfont.set_size(int(p_window_size[0]/30.))
            self.glfont.draw_text(p_window_size[0]/2.,p_window_size[1]/4.,'Touch {} more times to cancel calibration.'.format(self.clicks_to_close))

        glfwSwapBuffers(self._window)
        glfwMakeContextCurrent(active_window)