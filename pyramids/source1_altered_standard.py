import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import numpy as np
from imageio import imread

import scipy.ndimage as ndi

from skimage.morphology import disk, remove_small_objects, remove_small_holes, opening
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions for interactive plots
# ─────────────────────────────────────────────────────────────────────────────

def addPoint(path_collection, new_point, c='b'):
    """
    Add a new point to a PathCollection as produced by plt.scatter.
    """
    offsets = path_collection.get_offsets()
    facecls = path_collection.get_facecolors()
    if facecls.shape[0] == 1:
        facecls = np.repeat(facecls, offsets.shape[0], axis=0)

    offsets = np.concatenate([offsets, np.array(new_point, ndmin=2)])
    facecls = np.concatenate(
        [facecls, np.array(plt.matplotlib.colors.to_rgba(c), ndmin=2)]
    )

    path_collection.set_offsets(offsets)
    path_collection.set_facecolors(facecls)
    path_collection.axes.figure.canvas.draw_idle()


def rmPoint(path_collection, coordinates):
    """
    Remove the closest point to `coordinates` from a PathCollection as produced by plt.scatter.
    """
    offsets = path_collection.get_offsets()
    if offsets.size == 0:
        return

    facecls = path_collection.get_facecolors()
    if facecls.shape[0] == 1:
        facecls = np.repeat(facecls, offsets.shape[0], axis=0)

    idx_closest = np.argmin(np.sum((offsets - np.array(coordinates, ndmin=2))**2., axis=1))
    offsets = np.delete(offsets, idx_closest, axis=0)
    facecls = np.delete(facecls, idx_closest, axis=0)

    path_collection.set_offsets(offsets)
    path_collection.set_facecolors(facecls)
    path_collection.axes.figure.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
# Core classes
# ─────────────────────────────────────────────────────────────────────────────

class Image:
    def __init__(
        self,
        file_name=None,
        pixel_size=1.0,
        pixel_size_units='µm',
        scalebar_size=None,
        autocrop=True,
    ):
        if file_name:
            self.load(file_name, True)
        else:
            self.data = np.zeros([32, 32])

        self.set_scale(pixel_size, pixel_size_units)
        self.scalebar_size = scalebar_size

        if autocrop:
            self.data = self.data[:1785, :]

    def load(self, file_name, to_float=True):
        self.data = imread(file_name)
        if to_float:
            self.data = self.data / 2**8

    def set_scale(self, size, units='µm'):
        self._ps = size
        self._pu = units

    def show(self, fig=None, ax=None, colorbar=False, scalebar_size=None):
        if fig is None:
            fig, ax = plt.subplots()
        elif (fig is not None) and (ax is None):
            ax = fig.add_subplot()

        im = ax.imshow(self.data, cmap=plt.cm.afmhot)

        if colorbar:
            plt.colorbar(im, ax=ax)

        ax.axis('off')

        # Use manually provided scalebar_size first;
        # otherwise use the value stored in the Image object.
        size_to_use = scalebar_size if scalebar_size is not None else self.scalebar_size

        if size_to_use is not None:
            self._add_scalebar_artist(ax, size_to_use)

        return fig, ax

    def _add_scalebar_artist(self, axis, size_scaled):
        scalebar_size_pix = size_scaled / self._ps

        scalebar = AnchoredSizeBar(
            transform=axis.transData,
            size=scalebar_size_pix,
            label='{0:d} {1}'.format(int(round(size_scaled)), self._pu),
            loc='lower left',
            pad=1,
            color='white',
            frameon=False,
            label_top=True,
            size_vertical=8,
        )

        axis.add_artist(scalebar)

    def median_filter(self, *args, **kwargs):
        self.data = ndi.median_filter(self.data, *args, **kwargs)

    def gaussian_filter(self, *args, **kwargs):
        self.data = ndi.gaussian_filter(self.data, *args, **kwargs)


class FindMaxima:
    def __init__(self, image):
        if not isinstance(image, Image):
            raise AttributeError('The provided image attribute must be of Image type')
        self.image = image
        self.update()

    def update(self):
        self.auto_foreground()
        self.auto_fill_basins()

    def auto_foreground(self, threshold_factor=1.0, *args, **kwargs):
        if len(args) == 0:
            kwargs.setdefault('structure', disk(4))
            kwargs.setdefault('iterations', 6)

        self.foreground_threshold = threshold_otsu(self.image.data) * threshold_factor
        mask = self.image.data > self.foreground_threshold
        mask = ndi.binary_erosion(mask, *args, **kwargs)
        # mask[0, :] = 0
        # mask[:, 1] = 0
        self.foreground = ndi.binary_dilation(mask, *args, **kwargs)

    def auto_fill_basins(self, min_distance=1, *args, **kwargs):
        if not args and 'iterations' not in kwargs:
            kwargs['iterations'] = 4

        fg_labels, _ = ndi.label(self.foreground)
        coords = peak_local_max(
            image=self.image.data,
            labels=fg_labels,
            num_peaks_per_label=1,
            min_distance=min_distance,
            threshold_abs=0,
            threshold_rel=0,
            footprint=np.ones((3, 3), bool),
            exclude_border=False
        )
        marker_mask = np.zeros_like(self.image.data, dtype=bool)
        marker_mask[tuple(coords.T)] = True
        marker_mask = ndi.binary_dilation(marker_mask, *args, **kwargs)
        markers_img, self.Nlabels = ndi.label(marker_mask)
        self.markers = coords
        self.labels = watershed(-self.image.data, markers_img)


class PyramidTool:
    """
    Supports:
    - left click: add marker (blue)
    - right click: remove nearest marker

    Optional behavior:
    - if freeze_deleted_regions=True, removed regions are frozen as label 0
    - if live_measure=True, rays are re-measured after each click

    Also: 
    - prevents duplicates (exact same (y,x) in manual_markers)
    """

    def __init__(self, image, *args, live_measure=False, freeze_deleted_regions=False, **kwargs):
        self.image = image
        self.find_maxima = FindMaxima(self.image)

        # manual clicks tracked as (y,x)
        self.manual_markers = []

        # reversible "frozen holes": list of boolean masks (each is one excluded region)
        self.excluded_regions = []

        # remember last angle so clicking updates rays consistently
        self._last_pyramid_angle = 45.0

        # NEW
        self.live_measure = live_measure
        self.freeze_deleted_regions = freeze_deleted_regions

        self.interactive_plot(*args, **kwargs)

    # ───────── helpers for exclusions ─────────

    def _remove_exclusion_if_contains(self, y, x):
        """If (y,x) falls inside an excluded region, remove that exclusion."""
        if not self.excluded_regions:
            return
        h, w = self.image.data.shape
        if not (0 <= y < h and 0 <= x < w):
            return
        for i, m in enumerate(self.excluded_regions):
            if m[y, x]:
                self.excluded_regions.pop(i)
                return

    def _merge_or_append_exclusion(self, region_mask):
        """Merge overlapping exclusions; otherwise append."""
        for i, m in enumerate(self.excluded_regions):
            if np.any(m & region_mask):
                self.excluded_regions[i] = m | region_mask
                return
        self.excluded_regions.append(region_mask)

    def _apply_exclusions(self, labels_img):
        """Set excluded regions to label 0."""
        if not self.excluded_regions:
            return labels_img
        out = np.array(labels_img, copy=True)
        for m in self.excluded_regions:
            out[m] = 0
        return out

    def _exclude_region_at(self, y, x):
        """
        Freeze the CURRENT watershed region under (y,x) as excluded (label 0),
        without letting neighbors expand into it.
        """
        ws = np.array(self.watershed_image.get_array(), copy=True)
        h, w = ws.shape
        if not (0 <= y < h and 0 <= x < w):
            return

        lbl = int(ws[y, x])
        if lbl == 0:
            return

        region = (ws == lbl)
        self._merge_or_append_exclusion(region)

        # immediately show it as 0
        ws[region] = 0
        self.watershed_image.set_data(ws)
        self.watershed_image.set_clim(0, max(int(np.max(ws)), 1))
        self.axes[1].figure.canvas.draw_idle()

    # ───────── plotting ─────────

    def interactive_plot(self, *args, **kwargs):
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, *args, **kwargs)
        main_ax, ws_ax = axs

        # panel 1
        self.image.show(fig=fig, ax=main_ax)
        self.contours = main_ax.contour(
            np.arange(self.image.data.shape[1]),
            np.arange(self.image.data.shape[0]),
            self.find_maxima.foreground,
            [0.5],
            linewidths=0.5,
            colors='g',
        )

        self.pc_markers = main_ax.scatter(
            self.find_maxima.markers[:, 1],
            self.find_maxima.markers[:, 0],
            s=18,
            c='red',
            marker='o',
            edgecolors='white',
            linewidth=2.0,
            zorder=5,
        )

        # panel 2
        self.image.show(fig=fig, ax=ws_ax)
        ws_ax.images[0].set_cmap(plt.cm.gray)
        self.watershed_image = ws_ax.imshow(
            self.find_maxima.labels, cmap=plt.cm.flag, alpha=.2
        )

        def onClick(event):
            if event.inaxes is not main_ax:
                return

            if event.xdata is None or event.ydata is None:
                return

            y, x = int(round(event.ydata)), int(round(event.xdata))

            # LEFT CLICK: add marker
            if event.button == 1:
                if self.freeze_deleted_regions:
                    self._remove_exclusion_if_contains(y, x)

                # prevent duplicates in manual list
                if (y, x) not in self.manual_markers:
                    addPoint(self.pc_markers, (x, y), c='red')
                    self.manual_markers.append((y, x))
                    self.update_watershed()

                    # only re-measure if requested
                    if self.live_measure:
                        self.measure_pyramids(self._last_pyramid_angle)

            # RIGHT CLICK: remove nearest marker
            elif event.button == 3:
                offsets = self.pc_markers.get_offsets()
                if offsets.size == 0:
                    return

                # find nearest point in scatter (in x,y space)
                dx = offsets[:, 0] - event.xdata
                dy = offsets[:, 1] - event.ydata
                idx = int(np.argmin(dx*dx + dy*dy))
                px, py = offsets[idx, 0], offsets[idx, 1]  # scatter stores (x,y)

                # 1) remove it from scatter
                rmPoint(self.pc_markers, (px, py))

                # 2) if it was a manual marker, remove from manual list too
                py_i, px_i = int(round(py)), int(round(px))
                if self.manual_markers:
                    j = int(np.argmin([
                        (my - py)**2 + (mx - px)**2
                        for my, mx in self.manual_markers
                    ]))
                    # only remove if really close
                    if (self.manual_markers[j][0] - py_i)**2 + (self.manual_markers[j][1] - px_i)**2 <= 2:
                        self.manual_markers.pop(j)

                # 3) optionally freeze removed region
                if self.freeze_deleted_regions:
                    self._exclude_region_at(py_i, px_i)

                # 4) update watershed from remaining points
                self.update_watershed()

                # only re-measure if requested
                if self.live_measure:
                    self.measure_pyramids(self._last_pyramid_angle)

        fig.canvas.mpl_connect('button_press_event', onClick)
        fig.tight_layout()
        self.axes = axs

    @staticmethod
    def _clear_lines(ax):
        for ln in list(ax.lines):
            ln.remove()

    def update_watershed(self):
        offsets = self.pc_markers.get_offsets()

        if offsets.size == 0:
            labels_img = np.zeros_like(self.image.data, dtype=int)
        else:
            coords = np.round(offsets[:, ::-1]).astype(int)  # (y,x)
            h, w = self.image.data.shape
            coords[:, 0] = np.clip(coords[:, 0], 0, h - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, w - 1)

            mask = np.zeros_like(self.image.data, dtype=bool)
            mask[coords[:, 0], coords[:, 1]] = True
            mask = ndi.binary_dilation(mask, iterations=4)
            markers_img, _ = ndi.label(mask)
            labels_img = watershed(-self.image.data, markers_img)

        # If any marker currently exists inside an excluded region, un-exclude that region
        if self.freeze_deleted_regions and offsets.size != 0 and self.excluded_regions:
            coords = np.round(offsets[:, ::-1]).astype(int)  # (y,x)
            h, w = labels_img.shape
            coords[:, 0] = np.clip(coords[:, 0], 0, h - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, w - 1)

            keep = []
            for m in self.excluded_regions:
                inside = any(m[yy, xx] for yy, xx in coords)
                if not inside:
                    keep.append(m)
            self.excluded_regions = keep

        # apply remaining exclusions
        if self.freeze_deleted_regions:
            labels_img = self._apply_exclusions(labels_img)

        ax = self.axes[1]
        self._clear_lines(ax)
        self.watershed_image.set_data(labels_img)
        self.watershed_image.set_clim(0, max(int(np.max(labels_img)), 1))
        ax.figure.canvas.draw_idle()

    def measure_pyramids(self, pyramid_angle=45.0):
        self._last_pyramid_angle = pyramid_angle

        pa = np.deg2rad(pyramid_angle)
        angles = (np.arange(4) * (np.pi / 2) + pa) - np.pi

        ws = np.asarray(self.watershed_image.get_array())
        h, w = ws.shape

        # skip auto regions under manual clicks
        manual_labels = set()
        for (cy, cx) in self.manual_markers:
            if 0 <= cy < h and 0 <= cx < w:
                lbl = int(ws[cy, cx])
                if lbl != 0:
                    manual_labels.add(lbl)

        centers, vertices = [], []

        # AUTO: skip manual-labeled regions, skip border-touching, skip excluded (label 0)
        for prop in regionprops(ws):
            lbl = int(prop.label)
            if lbl == 0 or lbl in manual_labels:
                continue

            minr, minc, maxr, maxc = prop.bbox
            if minr == 0 or minc == 0 or maxr == h or maxc == w:
                continue

            # faster than copying the full image for every label
            coords = prop.coords
            vals = self.image.data[coords[:, 0], coords[:, 1]]
            cy, cx = coords[np.argmax(vals)]

            centers.append((cy, cx))
            verts = []
            for a in angles:
                dy, dx = np.sin(a), np.cos(a)
                t, ly, lx = 0, cy, cx
                while True:
                    iy = int(round(cy + dy * t))
                    ix = int(round(cx + dx * t))
                    if iy < 0 or iy >= h or ix < 0 or ix >= w or ws[iy, ix] != lbl:
                        break
                    ly, lx = iy, ix
                    t += 1
                verts.append((ly - cy, lx - cx))
            vertices.append(verts)

        # MANUAL rays: trace until boundary
        bd = find_boundaries(ws) if self.manual_markers else None
        for cy, cx in self.manual_markers:
            if not (0 <= cy < h and 0 <= cx < w):
                continue

            centers.append((cy, cx))
            verts = []
            for a in angles:
                dy, dx = np.sin(a), np.cos(a)
                t, ly, lx = 0, cy, cx
                while True:
                    iy = int(round(cy + dy * t))
                    ix = int(round(cx + dx * t))
                    if iy < 0 or iy >= h or ix < 0 or ix >= w or bd[iy, ix]:
                        break
                    ly, lx = iy, ix
                    t += 1
                verts.append((ly - cy, lx - cx))
            vertices.append(verts)

        self.pyramid_center_coordinates = np.array(centers)
        self.pyramid_vertex_coordinates = np.array(vertices)
        self.update_plot_with_measurements()

    def update_plot_with_measurements(self):
        ax = self.axes[1]
        self._clear_lines(ax)

        if not hasattr(self, "pyramid_center_coordinates") or not hasattr(self, "pyramid_vertex_coordinates"):
            self.axes[1].figure.canvas.draw_idle()
            return

        for (cy, cx), verts in zip(self.pyramid_center_coordinates, self.pyramid_vertex_coordinates):
            for dy, dx in verts:
                ax.add_line(
                    plt.Line2D(
                        [cx, cx + dx],
                        [cy, cy + dy],
                        linestyle='--',
                        linewidth=2,
                        color='darkblue'
                    )
                )
        ax.figure.canvas.draw_idle()

    def save_pyramid_measurements(self, file_name):
        manual_arr = np.array(self.manual_markers, dtype=int).reshape(-1, 2)
        np.savez(
            file_name,
            centers=self.pyramid_center_coordinates,
            vertices=self.pyramid_vertex_coordinates,
            manual_markers=manual_arr
        )

    def load_pyramid_measurements(self, file_name):
        data = np.load(file_name)
        self.pyramid_center_coordinates = data['centers']
        self.pyramid_vertex_coordinates = data['vertices']

        if 'manual_markers' in data:
            manual_arr = np.asarray(data['manual_markers'], dtype=int).reshape(-1, 2)
            self.manual_markers = [tuple(x) for x in manual_arr]
        else:
            self.manual_markers = []

        self.update_plot_with_measurements()