import logging
logger = logging.getLogger(__name__)

import cv2
import numpy as np

from surface_tracker.surface_tracker import Surface_Tracker
from surface_tracker.surface_online import Surface_Online
from surface_tracker.surface_marker import Surface_Marker
from surface_tracker.gui import Heatmap_Mode

import gl_utils
import pyglui
import pyglui.cygl.utils as pyglui_utils

def sortCorners(corners, center):
    """
    corners : list of points
    center : point
    """
    top = [corner for corner in corners if corner[1] < center[1]]
    bot = [corner for corner in corners if corner[1] >= center[1]]

    corners = np.zeros(shape=(4,2))

    if (len(top) == 2) and (len(bot) == 2):
        tl, tr = sorted(top, key=lambda p: p[0])
        bl, br = sorted(bot, key=lambda p: p[0])

    corners[0] = np.array(tl)
    corners[1] = np.array(tr)
    corners[2] = np.array(br)
    corners[3] = np.array(bl)

    return corners

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

def detect_screen_corners(gray_img, draw_contours=False):
    edges = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, -5)

    *_, contours, hierarchy = cv2.findContours(edges,
                                    mode=cv2.RETR_TREE,
                                    method=cv2.CHAIN_APPROX_SIMPLE,offset=(0,0)) #TC89_KCOS

    # if draw_contours:
    #     cv2.drawContours(gray_img, contours,-1, (0,0,0))

    # remove extra encapsulation
    hierarchy = hierarchy[0]
    contours = np.array(contours, dtype=object)

    # keep only contours                        with parents     and      children
    # contours = contours[np.logical_and(hierarchy[:,3]>=0, hierarchy[:,2]>=0)]
    # contours = np.array(contours)

    contours = np.array([c for c, h in zip(contours, hierarchy) if h[3] >= 0 and h[2] >= 0 and cv2.contourArea(c) > (20 * 2500)], dtype=object)

    screen_corners = []
    if len(contours) > 0:
        contours = contours[0].astype(np.int32)
        epsilon = cv2.arcLength(contours, True)*0.1
        aprox_contours = [cv2.approxPolyDP(contours, epsilon, True)]
        rect_cand = [r for r in aprox_contours if r.shape[0]==4]

        for count, r in enumerate(rect_cand):
            r = np.float32(r)
            cv2.cornerSubPix(gray_img, r, (3,3), (-1,-1), criteria)
            corners = np.array([r[0][0], r[1][0], r[2][0], r[3][0]])
            centroid = corners.sum(axis=0, dtype='float64')*0.25
            centroid.shape = (2)

            corners = sortCorners(corners, centroid)
            r[0][0], r[1][0], r[2][0], r[3][0] = corners[0], corners[1], corners[2], corners[3]

            corner = {'id':32+count,
                        'verts':r.tolist(),
                        'perimeter':cv2.arcLength(r,closed=True),
                        'centroid':centroid.tolist(),
                        "frames_since_true_detection":0,
                        "id_confidence":1.}
            screen_corners.append(corner)

    return map(Surface_Marker.from_square_tag_detection, screen_corners)

class Screen_Tracker_Online(Surface_Tracker):
    """
    The Screen_Tracker_Online does marker based AOI tracking in real-time. All
    necessary computation is done per frame.
    """
    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return "Screen Tracker"

    def __init__(self, g_pool, *args, **kwargs):
        self.freeze_scene = False
        self.frozen_scene_frame = None
        self.frozen_scene_tex = None
        super().__init__(g_pool, *args, use_online_detection=True, **kwargs)

        self.menu = None
        self.button = None
        self.add_button = None

    @property
    def Surface_Class(self):
        return Surface_Online

    @property
    def _save_dir(self):
        return self.g_pool.user_dir

    @property
    def has_freeze_feature(self):
        return True

    @property
    def supported_heatmap_modes(self):
        return [Heatmap_Mode.WITHIN_SURFACE]

    @property
    def ui_info_text(self):
        return (
            "This plugin detects and tracks a visible high contrast computer screen in the scene."
            "You can NOT define markers, the whole screen must be visible within the world view."
            "Homography to map the screen to a 2D plane is computed from the screen corners."
            "Marker directions are hard coded to be the corners of the screen."
        )

    def _update_markers(self, frame):
        self._detect_markers(frame)

    def _detect_markers(self, frame):
        self.markers = detect_screen_corners(gray_img=frame.gray)
        # markers = self._remove_duplicate_markers(markers)

    def _update_ui_custom(self):
        def set_freeze_scene(val):
            self.freeze_scene = val
            if val:
                self.frozen_scene_tex = pyglui_utils.Named_Texture()
                self.frozen_scene_tex.update_from_ndarray(self.current_frame.img)
            else:
                self.frozen_scene_tex = None

        self.menu.append(
            pyglui.ui.Switch(
                "freeze_scene", self, label="Freeze Scene", setter=set_freeze_scene
            )
        )

    def _per_surface_ui_custom(self, surface, surf_menu):
        def set_gaze_hist_len(val):
            if val <= 0:
                logger.warning("Gaze history length must be a positive number!")
                return
            surface.gaze_history_length = val

        surf_menu.append(
            pyglui.ui.Text_Input(
                "gaze_history_length",
                surface,
                label="Gaze History Length [seconds]",
                setter=set_gaze_hist_len,
            )
        )

    def recent_events(self, events):
        if self._ui_heatmap_mode_selector is not None:
            self._ui_heatmap_mode_selector.read_only = True
        if self.freeze_scene:
            # If frozen, we overwrite the frame event with the last frame we have saved
            current_frame = events.get("frame")
            events["frame"] = self.current_frame

        super().recent_events(events)

        if not self.current_frame:
            return

        self._update_surface_gaze_history(events, self.current_frame.timestamp)

        if self.gui.show_heatmap:
            self._update_surface_heatmaps()

        if self.freeze_scene:
            # After we are done, we put the actual current_frame back, so other
            # plugins can access it.
            events["frame"] = current_frame

    def _update_surface_locations(self, frame_index):
        for surface in self.surfaces:
            surface.update_location(frame_index, self.markers, self.camera_model)

    def _update_surface_corners(self):
        for surface, corner_idx in self._edit_surf_verts:
            if surface.detected:
                surface.move_corner(
                    corner_idx, self._last_mouse_pos.copy(), self.camera_model
                )

    def _update_surface_heatmaps(self):
        for surface in self.surfaces:
            gaze_on_surf = surface.gaze_history
            gaze_on_surf = (
                g
                for g in gaze_on_surf
                if g["confidence"] >= self.g_pool.min_data_confidence
            )
            gaze_on_surf = list(gaze_on_surf)
            surface.update_heatmap(gaze_on_surf)

    def _update_surface_gaze_history(self, events, world_timestamp):
        surfaces_gaze_dict = {
            e["name"]: e["gaze_on_surfaces"] for e in events["surfaces"]
        }

        for surface in self.surfaces:
            try:
                surface.update_gaze_history(
                    surfaces_gaze_dict[surface.name], world_timestamp
                )
            except KeyError:
                pass

    def on_add_surface_click(self, _=None):
        if self.freeze_scene:
            logger.warning("Surfaces cannot be added while the scene is frozen!")
        else:
            # NOTE: This is slightly different than the super() implementation.
            # We need to save the surface definition after adding it, but the Surface
            # Store does not store undefined surfaces. Therefore, we need to call
            # surface.update_location() once. This will define the surface and allow us
            # to save it.
            if self.markers and self.current_frame is not None:
                surface = self.Surface_Class(name=f"Surface {len(self.surfaces) + 1}")
                self.add_surface(surface)
                surface.update_location(
                    self.current_frame.index, self.markers, self.camera_model
                )
                self.save_surface_definitions_to_file()
            else:
                logger.warning(
                    "Can not add a new surface: No markers found in the image!"
                )

    def gl_display(self):
        if self.freeze_scene:
            self.gl_display_frozen_scene()
        super().gl_display()

    def gl_display_frozen_scene(self):
        gl_utils.clear_gl_screen()

        gl_utils.make_coord_system_norm_based()

        self.frozen_scene_tex.draw()

        gl_utils.make_coord_system_pixel_based(
            (self.g_pool.capture.frame_size[1], self.g_pool.capture.frame_size[0], 3)
        )

del Surface_Tracker
