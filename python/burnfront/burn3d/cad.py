"""Grain geometry from CAD: STEP (via OCP / cadquery-ocp) or STL (via trimesh).

Pipeline
--------
1. Tessellate (STEP: BRepMesh with user linear/angular deflection; OCP reports millimetres) or load
   an STL (units given by the user). Merge coincident vertices.
2. Validate: watertight, consistent winding, positive volume, single body; compare the mesh volume
   with the exact CAD volume (STEP).
3. Orient: user-selected axis -> +z, centred on the case, z from 0 to L.
4. Classify surfaces: planar end faces perpendicular to the axis at z = 0 or L are inhibited; the
   remaining lateral faces split into connected components; the component reaching the largest
   radius is the inert outer wall (case), the others are burning port surfaces.
5. Per slice, nodes inside the solid are found by even-odd ray crossing against the exact plane
   section of the mesh; non-solid nodes not connected to the slice border are port.
6. phi = exact signed distance to the port surface (negative in the port) on the whole fuel side;
   chi = exact signed distance to the outer wall (negative inside) near the wall.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from burnfront.burn3d.grid import Grid3D, Grain3D
from burnfront.burn3d.meshdist import MeshDistance

UNITS = {'m': 1.0, 'mm': 1e-3, 'cm': 1e-2, 'in': 0.0254, 'ft': 0.3048}


def _s(cls, name):
    """OCP exposes static methods with or without a trailing _s depending on version."""
    return getattr(cls, name + '_s', None) or getattr(cls, name)


# --- loading --------------------------------------------------------------------------------------

def tessellate_step(path, linear_deflection=0.01, angular_deflection=0.1, relative=False):
    """Tessellate a STEP file. Deflections are in the file's model units as converted by OCP (mm).

    Returns (vertices (N,3) in mm, faces (M,3), exact CAD volume in mm^3).
    """
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise IOError(f'could not read STEP file {path}')
    reader.TransferRoots()
    shape = reader.OneShape()
    props = GProp_GProps()
    _s(BRepGProp, 'VolumeProperties')(shape, props)
    cad_volume = abs(props.Mass())

    BRepMesh_IncrementalMesh(shape, float(linear_deflection), bool(relative), float(angular_deflection), True)
    to_face = _s(TopoDS, 'Face')
    triangulation = _s(BRep_Tool, 'Triangulation')
    V, F, off = [], [], 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = to_face(exp.Current())
        loc = TopLoc_Location()
        tri = triangulation(face, loc)
        if tri is not None:
            trsf = None if loc.IsIdentity() else loc.Transformation()
            pts = []
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i)
                if trsf is not None:
                    p = p.Transformed(trsf)
                pts.append((p.X(), p.Y(), p.Z()))
            tris = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)]) - 1
            if face.Orientation() == TopAbs_REVERSED:
                tris = tris[:, ::-1]
            V.append(np.array(pts))
            F.append(tris + off)
            off += len(pts)
        exp.Next()
    if not V:
        raise ValueError('STEP file has no triangulated faces')
    return np.concatenate(V), np.concatenate(F), cad_volume


def load_stl(path):
    import trimesh
    m = trimesh.load(str(path), force='mesh')
    return np.asarray(m.vertices), np.asarray(m.faces)


# --- validation / orientation -------------------------------------------------------------------

@dataclass
class MeshReport:
    n_vertices: int
    n_faces: int
    watertight: bool
    winding_consistent: bool
    n_bodies: int
    volume: float                 # m^3, from the closed mesh
    cad_volume: float = np.nan    # m^3, exact (STEP only)
    OD: float = np.nan
    L: float = np.nan
    outer_radius_spread: float = np.nan   # (max-min)/max radius of outer-wall vertices
    n_ports: int = 0
    port_area: float = np.nan     # m^2, initial burning surface (tessellated)
    warnings: list = field(default_factory=list)

    @property
    def ok(self):
        return self.watertight and self.winding_consistent and self.n_bodies == 1 and self.volume > 0

    def __str__(self):
        lines = [f'mesh: {self.n_vertices} vertices, {self.n_faces} triangles, {self.n_bodies} body(ies)',
                 f'  watertight={self.watertight}  winding consistent={self.winding_consistent}',
                 f'  volume={self.volume*1e6:.4f} cm^3' + (f'  (CAD {self.cad_volume*1e6:.4f} cm^3, '
                 f'tessellation error {self.volume/self.cad_volume-1:+.2e})' if np.isfinite(self.cad_volume) else ''),
                 f'  OD={self.OD*1e3:.3f} mm  L={self.L*1e3:.3f} mm  outer-wall roundness spread={self.outer_radius_spread:.1e}',
                 f'  ports={self.n_ports}  initial burning surface={self.port_area*1e4:.3f} cm^2']
        lines += [f'  WARNING: {w}' for w in self.warnings]
        return '\n'.join(lines)


def _rotation_to_z(axis):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, z)
    c = float(a @ z)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1/(1 + c))


def _axis_vector(axis):
    if isinstance(axis, str):
        sgn = -1.0 if axis.startswith('-') else 1.0
        return sgn*np.eye(3)['xyz'.index(axis.lstrip('+-').lower())]
    return np.asarray(axis, dtype=float)


def prepare_mesh(V, F, scale=1.0, axis='z', cad_volume=np.nan):
    """Scale to metres, merge vertices, validate, orient the axis to +z and centre. Returns
    (trimesh.Trimesh, MeshReport, face classes) with classes 0 = end, 1 = outer wall, 2 = port."""
    import trimesh
    m = trimesh.Trimesh(np.asarray(V, dtype=float)*scale, np.asarray(F), process=True)
    m.merge_vertices()
    warnings = []
    if m.is_winding_consistent and m.is_watertight and m.volume < 0:
        m.invert()
        warnings.append('mesh was inside-out; normals inverted')
    R = _rotation_to_z(_axis_vector(axis))
    m.apply_transform(np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))
    lo, hi = m.bounds
    m.apply_translation([-(lo[0] + hi[0])/2, -(lo[1] + hi[1])/2, -lo[2]])
    L = float(m.bounds[1][2])

    # Face classes
    n = m.face_normals
    cz = m.triangles_center[:, 2]
    tol = 1e-6*max(L, 1e-9)
    end = (np.abs(n[:, 2]) > 0.999) & ((cz < tol) | (cz > L - tol))
    lat = np.nonzero(~end)[0]
    adj = m.face_adjacency
    keep = (~end[adj[:, 0]]) & (~end[adj[:, 1]])
    adj = adj[keep]
    G = coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(len(m.faces),)*2)
    ncomp, lab = connected_components(G, directed=False)
    rc = np.hypot(m.triangles_center[:, 0], m.triangles_center[:, 1])
    outer_lab = lab[lat[np.argmax(rc[lat])]]
    cls = np.where(end, 0, np.where(lab == outer_lab, 1, 2))
    port_labels = np.unique(lab[cls == 2])

    outer_v = np.unique(m.faces[cls == 1])
    rv = np.hypot(m.vertices[outer_v, 0], m.vertices[outer_v, 1])
    body_count = len(m.split(only_watertight=False)) if len(m.faces) < 2_000_000 else 1
    rep = MeshReport(
        n_vertices=len(m.vertices), n_faces=len(m.faces), watertight=bool(m.is_watertight),
        winding_consistent=bool(m.is_winding_consistent), n_bodies=body_count,
        volume=float(abs(m.volume)) if m.is_watertight else np.nan,
        cad_volume=cad_volume*scale**3 if np.isfinite(cad_volume) else np.nan,
        OD=2*float(rv.max()), L=L, outer_radius_spread=float((rv.max() - rv.min())/rv.max()),
        n_ports=int(len(port_labels)), port_area=float(m.area_faces[cls == 2].sum()), warnings=warnings,
    )
    if not rep.watertight:
        rep.warnings.append('mesh is not watertight: inside/outside classification may be wrong')
    if not rep.winding_consistent:
        rep.warnings.append('inconsistent face winding')
    if rep.n_bodies != 1:
        rep.warnings.append(f'{rep.n_bodies} separate bodies')
    if rep.n_ports == 0:
        rep.warnings.append('no port surface found (no lateral faces besides the outer wall)')
    if rep.outer_radius_spread > 0.02:
        rep.warnings.append('outer wall is not a circular cylinder; envelope taken from the mesh as is')
    return m, rep, cls


# --- voxelization ---------------------------------------------------------------------------------

def _slice_segments(mesh, z):
    import trimesh
    seg = trimesh.intersections.mesh_plane(mesh, plane_normal=[0, 0, 1], plane_origin=[0, 0, z])
    return np.asarray(seg)[:, :, :2]


def _inside_parity(X, Y, seg, chunk=2_000_000):
    """Even-odd test of points (X, Y) against closed section segments (half-open crossing rule)."""
    px, py = X.ravel(), Y.ravel()
    inside = np.zeros(px.size, dtype=bool)
    if len(seg) == 0:
        return inside.reshape(X.shape)
    x1, y1, x2, y2 = seg[:, 0, 0], seg[:, 0, 1], seg[:, 1, 0], seg[:, 1, 1]
    step = max(1, chunk // len(seg))
    for s0 in range(0, px.size, step):
        X_, Y_ = px[s0:s0 + step, None], py[s0:s0 + step, None]
        crosses = (y1 <= Y_) != (y2 <= Y_)
        t = (Y_ - y1) / np.where(y2 != y1, y2 - y1, 1.0)
        xi = x1 + t*(x2 - x1)
        inside[s0:s0 + step] = (np.sum(crosses & (X_ < xi), axis=1) % 2) == 1
    return inside.reshape(X.shape)


def _subdivided(mesh, faces_mask, max_edge):
    """Triangle soup covering the selected faces with pieces of size ~max_edge (conformity is not
    needed for distance queries).

    Each triangle is cut by a structured k x m grid: k strips parallel to its shortest edge (apex =
    opposite vertex, k = ceil(longest side / max_edge)) and m divisions across (m = ceil(shortest
    edge / max_edge)). This keeps long extrusion facets (e.g. 300 x 0.7 mm cylinder strips or
    15 x 300 mm planar faces) as compact pieces; uniform or longest-edge bisection would turn them
    into hundreds of thousands of slivers.
    """
    T = mesh.vertices[mesh.faces[faces_mask]]
    n = len(T)
    e = np.stack([np.linalg.norm(T[:, 2] - T[:, 1], axis=1), np.linalg.norm(T[:, 0] - T[:, 2], axis=1),
                  np.linalg.norm(T[:, 1] - T[:, 0], axis=1)], axis=1)      # edge i is opposite vertex i
    j = np.argmin(e, axis=1)
    r = np.arange(n)
    p, u, v = T[r, j], T[r, (j + 1) % 3], T[r, (j + 2) % 3]
    k = np.maximum(1, np.ceil(np.maximum(np.linalg.norm(p - u, axis=1), np.linalg.norm(p - v, axis=1))/max_edge)).astype(int)
    m = np.maximum(1, np.ceil(e[r, j]/max_edge)).astype(int)
    out = []
    for kk, mm in np.unique(np.stack([k, m], 1), axis=0):
        sel = (k == kk) & (m == mm)
        t = (np.arange(kk + 1)/kk)[None, :, None]
        P = u[sel, None] + t*(p[sel, None] - u[sel, None])                      # (n, kk+1, 3)
        Q = v[sel, None] + t*(p[sel, None] - v[sel, None])
        q = (np.arange(mm + 1)/mm)[None, None, :, None]
        R = P[:, :, None] + q*(Q - P)[:, :, None]                               # (n, kk+1, mm+1, 3)
        a, b = R[:, :-1, :-1], R[:, :-1, 1:]
        c, d = R[:, 1:, 1:], R[:, 1:, :-1]
        out += [np.stack([a, b, c], -2).reshape(-1, 3, 3), np.stack([a, c, d], -2).reshape(-1, 3, 3)]
    T = np.concatenate(out)
    area2 = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    T = T[area2 > 1e-12*max_edge**2]                     # drop pieces collapsed at the apex
    return T.reshape(-1, 3), np.arange(3*len(T)).reshape(-1, 3)


def grain_from_mesh(mesh, report, cls, n_xy=96, n_z=48, margin=4, port_band_cells=4, wall_band_cells=4,
                    verbose=False):
    """Build a Grain3D from an oriented, classified mesh (see prepare_mesh)."""
    OD, L = report.OD, report.L
    grid = Grid3D.cylinder(OD, L, n_xy, n_z, margin)
    hx, hy, hz = grid.h
    h = min(hx, hy)
    X, Y = grid.xy_mesh()
    nz = grid.z.size
    eps = 1e-6*L
    solid = np.zeros(grid.shape, dtype=bool)
    port = np.zeros(grid.shape, dtype=bool)
    for k, zk in enumerate(grid.z):
        seg = _slice_segments(mesh, float(np.clip(zk, eps, L - eps)))
        s = _inside_parity(X, Y, seg)
        lab, nl = ndimage.label(~s)
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        ext = np.isin(lab, border[border > 0])
        solid[:, :, k] = s
        port[:, :, k] = (~s) & (~ext)
    envelope = solid | port

    # Port-side band (cells) and envelope bands from per-slice Euclidean distance transforms
    dist_in_port = np.stack([ndimage.distance_transform_edt(port[:, :, k]) for k in range(nz)], axis=2)
    # Cells from the envelope boundary, on either side (edt measures the distance to the other set)
    dist_to_wall = np.stack([np.where(envelope[:, :, k], ndimage.distance_transform_edt(envelope[:, :, k]),
                                      ndimage.distance_transform_edt(~envelope[:, :, k])) for k in range(nz)], axis=2)

    # phi: exact distance to the port surface on the fuel side (+ thin port band)
    Vp, Fp = _subdivided(mesh, cls == 2, max_edge=h)
    md = MeshDistance(Vp, Fp)
    # Exact on: all fuel, a thin port-side band (cut cells), and both sides of the wall band (cells
    # cut by the wall need the true phi at exterior corners once the front approaches burn-through)
    mask = solid | (port & (dist_in_port <= port_band_cells)) | (dist_to_wall <= 3)
    ii, jj, kk = np.nonzero(mask)
    d = md.unsigned(np.stack([grid.x[ii], grid.y[jj], grid.z[kk]], axis=1))
    phi = np.where(port, -(port_band_cells + 2)*h, OD)   # placeholders: only the sign is used
    phi[ii, jj, kk] = np.where(port[ii, jj, kk], -d, d)

    # chi: exact distance to the outer wall near it, sign from the envelope
    Vw, Fw = _subdivided(mesh, cls == 1, max_edge=h)
    mdw = MeshDistance(Vw, Fw)
    wmask = dist_to_wall <= wall_band_cells
    ii, jj, kk = np.nonzero(wmask)
    dw = mdw.unsigned(np.stack([grid.x[ii], grid.y[jj], grid.z[kk]], axis=1))
    chi = np.where(envelope, -1.0, 1.0) * (wall_band_cells + 2) * h
    chi[ii, jj, kk] = np.where(envelope[ii, jj, kk], -dw, dw)
    if verbose:
        print(f'grid {grid.shape}: {mask.sum()} exact port distances, {wmask.sum()} wall distances')
    return Grain3D(grid, phi, chi)


def load_grain(path, units=None, axis='z', n_xy=96, n_z=48, linear_deflection=None,
               angular_deflection=0.1, verbose=True):
    """Load a STEP (.step/.stp) or STL (.stl) grain and build a Grain3D.

    units: model units of the file ('mm', 'm', 'cm', 'in', 'ft'). STEP files are converted to mm by
           OCP, so the default is 'mm' for both formats.
    axis: port axis in model coordinates ('x', 'y', 'z', optionally with '-', or a 3-vector); the
          head (injector) end is at the low end of this axis.
    linear_deflection: STEP tessellation chord tolerance in mm (default 1e-5 x largest part extent).
    Returns (Grain3D, MeshReport).
    """
    path = str(path)
    scale = UNITS[(units or 'mm').lower()]
    cad_volume = np.nan
    if path.lower().endswith(('.step', '.stp')):
        if linear_deflection is None:
            # Chord error of a curved wall of radius R changes the volume by ~(4/3) delta/R; tie the
            # default to the size of the part so tessellation error (~3e-5) stays far below grid error
            V0, F0, _ = tessellate_step(path, 1.0, 0.5)
            linear_deflection = 1e-5*float(np.max(np.ptp(V0, axis=0)))
        V, F, cad_volume = tessellate_step(path, linear_deflection, angular_deflection)
        scale = 1e-3 if units is None else scale
    elif path.lower().endswith('.stl'):
        V, F = load_stl(path)
    else:
        raise ValueError('expected a .step/.stp or .stl file')
    mesh, rep, cls = prepare_mesh(V, F, scale=scale, axis=axis, cad_volume=cad_volume)
    if verbose:
        print(rep)
    if not rep.ok:
        raise ValueError('mesh failed validation:\n' + str(rep))
    return grain_from_mesh(mesh, rep, cls, n_xy=n_xy, n_z=n_z), rep


# --- sample geometry ------------------------------------------------------------------------------

def write_star_grain_step(path, OD, L, n_tips, r_valley, r_tip, twist_turns=0.0, n_sections=None,
                          units='mm'):
    """Write a STEP file of a cylindrical grain (outer diameter OD, length L) with a star port,
    optionally twisted by ``twist_turns`` revolutions over L (ruled loft through n_sections stars).
    Dimensions are given in ``units``; the file is written in millimetres. Useful for tests/demos.
    """
    from OCP.gp import gp_Pnt, gp_Ax2, gp_Dir, gp_Vec
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone
    from burnfront.burn3d.shapes import star_vertices

    k = UNITS[units] / 1e-3
    OD, L, r_valley, r_tip = OD*k, L*k, r_valley*k, r_tip*k
    ext = 0.01*L  # let the cutting tool overhang both end faces

    def star_wire(z, rot):
        poly = BRepBuilderAPI_MakePolygon()
        for x, y in star_vertices(n_tips, r_valley, r_tip, rot):
            poly.Add(gp_Pnt(float(x), float(y), float(z)))
        poly.Close()
        return poly.Wire()

    if twist_turns == 0.0:
        face = BRepBuilderAPI_MakeFace(star_wire(-ext, 0.0)).Face()
        tool = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, L + 2*ext)).Shape()
    else:
        n_sections = n_sections or max(16, int(64*abs(twist_turns)))
        loft = BRepOffsetAPI_ThruSections(True, True)  # solid, ruled
        for z in np.linspace(-ext, L + ext, n_sections):
            loft.AddWire(star_wire(z, 2*np.pi*twist_turns*z/L))
        loft.Build()
        tool = loft.Shape()
    cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), OD/2, L).Shape()
    grain = BRepAlgoAPI_Cut(cyl, tool).Shape()
    w = STEPControl_Writer()
    w.Transfer(grain, STEPControl_AsIs)
    if w.Write(str(path)) != IFSelect_RetDone:
        raise IOError(f'could not write {path}')
    return path
