"""Voxel grid and grain geometry container."""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from burnfront.burn3d import measure
from burnfront.burn3d.levelset import reinitialize


@dataclass
class Grid3D:
    x: np.ndarray  # (nx,) node coordinates
    y: np.ndarray  # (ny,)
    z: np.ndarray  # (nz,) axial stations; z[0] = head end

    @property
    def h(self):
        return (self.x[1] - self.x[0], self.y[1] - self.y[0], self.z[1] - self.z[0])

    @property
    def shape(self):
        return (self.x.size, self.y.size, self.z.size)

    def xy_mesh(self):
        return np.meshgrid(self.x, self.y, indexing='ij')

    @classmethod
    def cylinder(cls, OD, L, n_xy, n_z, margin=4, z0=0.0):
        """Grid for a grain of outer diameter OD and length L.

        n_xy: cells across OD (spacing h = OD/n_xy); nodes symmetric about the axis with ``margin``
        extra cells beyond the case. n_z: number of axial stations (nodes) spanning [z0, z0+L].
        """
        h = OD / n_xy
        half = int(np.ceil(n_xy/2)) + margin
        x = np.arange(-half, half + 1) * h
        z = np.linspace(z0, z0 + L, n_z)
        return cls(x=x, y=x.copy(), z=z)


class Grain3D:
    """Burning-front level set ``phi`` plus static inert envelope ``chi`` on a Grid3D.

    ``phi`` must be an accurate signed distance within a band of ~8 cells around the front (see
    shapes.port_sdf / meshdist); values further away are never used by the evolution or the measures.
    reinit=True applies fast-marching reinitialization, which is only first-order accurate next to
    the front: use it for rough inputs, not for accuracy-critical fields.
    """

    def __init__(self, grid, phi, chi, reinit=False):
        phi = np.asarray(phi, dtype=float)
        if phi.shape != grid.shape:
            raise ValueError(f'phi shape {phi.shape} != grid shape {grid.shape}')
        chi = np.asarray(chi, dtype=float)
        if chi.ndim == 2:
            chi = chi[:, :, None]
        self.grid = grid
        self.phi = reinitialize(phi, grid.h) if reinit else phi
        self.chi = chi
        self.w = measure.trapz_weights(grid.z)
        hx, hy, _ = grid.h
        self.A_env = measure.slice_areas(np.broadcast_to(chi, grid.shape), hx, hy)

    # --- measures -------------------------------------------------------------------------------
    @property
    def L(self):
        return float(self.grid.z[-1] - self.grid.z[0])

    def table(self, levels, phi=None):
        hx, hy, _ = self.grid.h
        return measure.port_area_table(self.phi if phi is None else phi, self.chi, levels, hx, hy)

    def port_areas(self):
        return self.table([0.0])[:, 0]

    def fuel_volume(self, A_port=None):
        A_port = self.port_areas() if A_port is None else A_port
        return float(np.sum(self.w * (self.A_env - A_port)))

    def web_remaining(self):
        """Per-station minimum distance from the burning front to the inert envelope (m).

        Samples phi (trilinear) at the projections onto {chi = 0} of nodes near the envelope.
        Negative once the front has reached the envelope (burn-through) at that station.
        """
        g = self.grid
        h = g.h
        chi = np.broadcast_to(self.chi, g.shape)
        gx, gy = np.gradient(chi[:, :, 0] if self.chi.shape[2] == 1 else chi, h[0], h[1], axis=(0, 1))
        near = np.abs(chi) < 1.5*max(h[0], h[1])
        if self.chi.shape[2] == 1:
            gx = np.broadcast_to(gx[:, :, None], g.shape)
            gy = np.broadcast_to(gy[:, :, None], g.shape)
        ii, jj, kk = np.nonzero(near)
        c = chi[near]
        gn = np.hypot(gx[near], gy[near])
        gn = np.where(gn > 0, gn, 1.0)
        # Projection onto chi = 0 in index space
        pi = ii - c*gx[near]/gn/h[0]
        pj = jj - c*gy[near]/gn/h[1]
        vals = ndimage.map_coordinates(self.phi, [pi, pj, kk.astype(float)], order=1, mode='nearest')
        web = np.full(g.shape[2], np.inf)
        np.minimum.at(web, kk, vals)
        return web
