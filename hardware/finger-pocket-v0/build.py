"""Build a provisional enclosure from dimensions.json. Units: millimetres.

Requires numpy, manifold3d and trimesh. Produces the same CSG geometry as both
STL and OpenSCAD, plus a mesh verification report. No measured-fit claim.
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import manifold3d as m
import trimesh

ROOT = Path(__file__).resolve().parent


class Solid:
    def __init__(self, mesh, code):
        self.mesh, self.code = mesh, code

    def move(self, xyz):
        return Solid(self.mesh.translate(xyz), f"translate({json.dumps(xyz)}) {{\n{self.code}\n}}")

    def rotate(self, xyz):
        return Solid(self.mesh.rotate(xyz), f"rotate({json.dumps(xyz)}) {{\n{self.code}\n}}")

    def __add__(self, other):
        return Solid(self.mesh + other.mesh, f"union() {{\n{self.code}\n{other.code}\n}}")

    def __sub__(self, other):
        return Solid(self.mesh - other.mesh, f"difference() {{\n{self.code}\n{other.code}\n}}")


def box(size, at=(0, 0, 0)):
    assert min(size) > 0, size
    return Solid(m.Manifold.cube(size), f"cube({json.dumps(size)});").move(list(at))


def cylinder(h, r, at=(0, 0, 0), segments=48):
    return Solid(m.Manifold.cylinder(h, r, circular_segments=segments),
                 f"cylinder(h={h},r={r},$fn={segments});").move(list(at))


def union(items):
    result = items[0]
    for item in items[1:]:
        result = result + item
    return result


def hull(items):
    u = union(items)
    return Solid(u.mesh.hull(), f"hull() {{\n{u.code}\n}}")


def rounded_box(size, radius=1, at=(0, 0, 0)):
    x, y, z = size
    return hull([cylinder(z, radius, (i, j, 0), 24)
                 for i in (radius, x-radius) for j in (radius, y-radius)]).move(list(at))


def mesh_of(solid):
    mesh = solid.mesh.to_mesh()
    return trimesh.Trimesh(vertices=np.array(mesh.vert_properties)[:, :3],
                           faces=np.array(mesh.tri_verts), process=True)


def build(p):
    wall, floor, lid_t = p['wall'], p['base_floor'], p['lid_thickness']
    fw, fh, fd = p['finger_width'], p['finger_height'], p['finger_depth']
    fz, dt, gap = p['finger_floor_height'], p['deck_thickness'], p['deck_clearance_per_side']
    c, margin = p['breadboard_clearance_per_side'], p['end_service_margin']
    length = max(p['breadboard_length']+2*c+2*margin+2*wall, fd+2*wall+18)
    width = 2*wall+fw+wall+p['breadboard_width']+2*c
    height = math.ceil(max(floor+p['assembled_electronics_height']+p['wire_headroom'], fz+fh))
    by = 2*wall+fw+c
    bx = wall+margin+c
    sx, sy = p['sensor_distance_from_entrance'], wall+fw/2
    rear = wall+fd
    deck_z = fz-dt
    sensor_top = fz-p['sensor_package_height_above_pcb']
    sensor_bottom = sensor_top-p['sensor_pcb_thickness']
    assert floor < sensor_bottom < deck_z < fz
    assert deck_z <= sensor_top < fz
    assert p['sensor_pcb_width']+2*gap < fw-5
    assert sx-p['sensor_pcb_length']/2 > wall+2
    assert sx+p['sensor_pcb_length']/2 < rear-2
    assert 0 < p['sensor_aperture_width'] < p['sensor_pcb_width']-4
    assert 0 < p['sensor_aperture_length'] < p['sensor_pcb_length']-4
    eps = 0.05

    # Open tray. Front finger doorway runs up to the rim, avoiding a printed bridge.
    base = rounded_box([length, width, height], 2)
    base -= box([length-2*wall, width-2*wall, height], [wall, wall, floor])
    base += box([length, wall, height], [0, wall+fw, 0])
    base += box([wall, fw+wall, height], [rear, 0, 0])
    base -= box([wall+2*eps, fw, height], [-eps, wall, deck_z])
    # Flared mouth: one millimetre bevel along entrance sides and lower edge.
    base -= hull([box([eps, fw+2, height], [-eps, wall-1, deck_z-1]),
                  box([eps, fw, height], [wall+eps, wall, deck_z])])

    # A deliberately generous provisional USB opening on the opposite end.
    usb_y = by+p['breadboard_width']/2-p['usb_opening_width']/2
    base -= box([wall+2*eps, p['usb_opening_width'], p['usb_opening_height']],
                [length-wall-eps, usb_y, p['usb_opening_bottom']])
    # Wiring crosses the divider BELOW the finger deck.
    base -= box([12, wall+2*eps, 5], [sx-6, wall+fw-eps, floor+0.5])

    # Deck side ledges and four module-edge supports; component centres stay clear.
    for yy in (wall, wall+fw-2.5):
        base += box([fd, 2.5, deck_z-floor], [wall, yy, floor])
    deck_keys = [(12, wall), (rear-12, wall+fw-2.5)]
    for xx, yy in deck_keys:
        base += box([3, 2.5, 1.5+eps], [xx, yy, deck_z-eps])
    for xx in (sx-p['sensor_pcb_length']/2, sx+p['sensor_pcb_length']/2-3):
        for yy in (sy-p['sensor_pcb_width']/2, sy+p['sensor_pcb_width']/2-2):
            base += box([3, 2, sensor_bottom-floor+eps], [xx, yy, floor-eps])
    # Continue the wire passage through the deck ledge added above.
    base -= box([12, wall+2.5+2*eps, 5], [sx-6, wall+fw-2.5-eps, floor+0.5])

    # Screw posts stay outside both the breadboard and finger insertion volume.
    screws = [(6, width-8), (length-6, width-8),
              (rear+8, 7), (length-6, 7)]
    for xx, yy in screws:
        base += cylinder(height, 4.5, [xx, yy, 0])
        base -= cylinder(height+2*eps, p['screw_clearance_diameter']/2, [xx, yy, -eps])
        base -= cylinder(p['nut_pocket_depth']+eps, p['nut_across_flats']/math.sqrt(3),
                         [xx, yy, -eps], 6)

    # Flat removable deck; underside recess locates the provisional sensor PCB.
    deck = box([rear-2*gap, fw-2*gap, dt], [gap, wall+gap, deck_z])
    for xx, yy in deck_keys:
        deck -= box([3+2*gap, 2.5+2*gap, dt+2*eps], [xx-gap, yy-gap, deck_z-eps])
    aperture = rounded_box([p['sensor_aperture_length'], p['sensor_aperture_width'], dt+2*eps],
                           1, [sx-p['sensor_aperture_length']/2,
                               sy-p['sensor_aperture_width']/2, deck_z-eps])
    deck -= aperture
    deck -= box([p['sensor_pcb_length']+2*gap, p['sensor_pcb_width']+2*gap,
                 sensor_top-deck_z+eps],
                [sx-p['sensor_pcb_length']/2-gap, sy-p['sensor_pcb_width']/2-gap, deck_z-eps])
    # Small tapered upper edge at the optical aperture.
    deck -= hull([rounded_box([p['sensor_aperture_length'], p['sensor_aperture_width'], eps],
                              1, [sx-p['sensor_aperture_length']/2,
                                  sy-p['sensor_aperture_width']/2, fz-0.5]),
                  rounded_box([p['sensor_aperture_length']+1, p['sensor_aperture_width']+1, eps],
                              1.5, [sx-p['sensor_aperture_length']/2-0.5,
                                    sy-p['sensor_aperture_width']/2-0.5, fz+eps])])

    lid = rounded_box([length, width, lid_t], 2, [0, 0, height])
    roof_drop = height-(fz+fh)
    if roof_drop > 0:
        lid += box([rear-gap, fw-2*gap, roof_drop+eps], [0, wall+gap, fz+fh])
    # These feet retain the deck on its ledges when the lid is screwed down.
    for yy in (wall+gap, wall+fw-gap-1.5):
        lid += box([3, 1.5, height-fz+eps], [rear-gap-3, yy, fz])
    for xx, yy in screws:
        lid -= cylinder(lid_t+2*eps, p['screw_clearance_diameter']/2, [xx, yy, height-eps])

    # Small proof piece: entrance and sensor area only, for an inexpensive first fit.
    fit_base = Solid(base.mesh ^ box([rear+wall, fw+2*wall, height]).mesh,
                     f"intersection() {{\n{base.code}\n{box([rear+wall, fw+2*wall, height]).code}\n}}")
    fit_roof_cut = box([rear+wall, fw+2*wall, height+lid_t+1])
    fit_roof = Solid(lid.mesh ^ fit_roof_cut.mesh,
                     f"intersection() {{\n{lid.code}\n{fit_roof_cut.code}\n}}")
    assemblies = {'base': base, 'lid': lid, 'finger-deck': deck}
    printable = {
        'base': base,
        'lid': lid.rotate([180, 0, 0]).move([0, width, height+lid_t]),
        'finger-deck': deck.rotate([180, 0, 0]).move([-gap, wall+fw-gap, fz]),
        'finger-fit-sample': fit_base,
        'finger-fit-roof': fit_roof.rotate([180, 0, 0]).move([0, fw+2*wall, height+lid_t]),
    }
    reference = {
        'breadboard': box([p['breadboard_length'], p['breadboard_width'], 9.3], [bx, by, floor]),
        'controller': box([45, 18, 5], [bx+5, by+10, floor+12]),
        'sensor': box([p['sensor_pcb_length'], p['sensor_pcb_width'], p['sensor_pcb_thickness']],
                      [sx-p['sensor_pcb_length']/2, sy-p['sensor_pcb_width']/2, sensor_bottom]),
        'optics': box([5.6, 3.3, p['sensor_package_height_above_pcb']],
                      [sx-2.8, sy-1.65, sensor_top])
    }
    return assemblies, printable, reference, {
        'outer_size_mm': [length, width, height+lid_t],
        'finger_clear_size_mm': [fd-3-gap, fw-2*gap, fh],
        'breadboard_origin_mm': [bx, by, floor],
        'breadboard_allowed_mm': [p['breadboard_length'], p['breadboard_width']],
        'body_height_mm': height,
        'status': 'Geometry checked only; all hardware, cable and finger fits remain unverified.'
    }


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--preview-data', type=Path)
    args = args.parse_args()
    p = json.loads((ROOT/'dimensions.json').read_text())
    assembly, printable, refs, info = build(p)
    for name, solid in printable.items():
        mesh = mesh_of(solid)
        assert str(solid.mesh.status()).endswith('NoError'), (name, solid.mesh.status())
        assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0, name
        assert len(solid.mesh.decompose()) == 1, (name, 'disconnected geometry')
        assert mesh.bounds[0, 2] > -0.001, (name, 'below print bed')
        mesh.export(ROOT/f'{name}.stl')
        (ROOT/f'{name}.scad').write_text('// Concept v0; mm. Regenerate using dimensions.json and build.py.\n'+solid.code+'\n')
        info[name] = {'watertight': bool(mesh.is_watertight), 'volume_cm3': round(float(mesh.volume)/1000, 2),
                      'size_mm': np.round(mesh.extents, 3).tolist(), 'triangles': len(mesh.faces)}
    # Independent assembly checks catch solid interference and reserved-volume clashes.
    for a, b in [('base', 'lid'), ('base', 'finger-deck'), ('lid', 'finger-deck')]:
        overlap = (assembly[a].mesh ^ assembly[b].mesh).volume()
        assert overlap < 0.001, (a, b, overlap)
    for name in ('base', 'lid', 'finger-deck'):
        overlap = (assembly[name].mesh ^ refs['breadboard'].mesh).volume()
        assert overlap < 0.001, ('breadboard collision', name, overlap)
        for ref_name in ('sensor', 'optics'):
            overlap = (assembly[name].mesh ^ refs[ref_name].mesh).volume()
            assert overlap < 0.001, (ref_name, 'collision', name, overlap)
    finger_space = box(info['finger_clear_size_mm'],
                       [p['wall'], p['wall']+p['deck_clearance_per_side'], p['finger_floor_height']])
    for name in ('base', 'lid', 'finger-deck'):
        assert (assembly[name].mesh ^ finger_space.mesh).volume() < 0.001, ('finger obstruction', name)
    length, width, _ = info['outer_size_mm']
    usb_y = info['breadboard_origin_mm'][1]+p['breadboard_width']/2-p['usb_opening_width']/2
    usb_plug_space = box([p['wall']+12, p['usb_opening_width'], p['usb_opening_height']],
                         [length-p['wall']-12, usb_y, p['usb_opening_bottom']])
    assert (assembly['base'].mesh ^ usb_plug_space.mesh).volume() < 0.001, 'USB plug obstructed'
    info['assembly_interference_check'] = 'passed for printed parts, provisional breadboard/sensor volumes, clear finger pocket and USB plug access'
    (ROOT/'verification.json').write_text(json.dumps(info, indent=2)+'\n')
    (ROOT/'assembly.scad').write_text('// Assembly preview, mm. Hardware shown only as references.\n'+
        '\n'.join(f'color("{color}") {{\n{solid.code}\n}}' for solid, color in
                  [(assembly['base'], 'Gainsboro'), (assembly['finger-deck'], 'Silver'),
                   (assembly['lid'].move([0,0,20]), 'LightGray'), (refs['breadboard'], 'White'),
                   (refs['controller'], 'SeaGreen'), (refs['sensor'], 'SeaGreen')])+'\n')
    data = {'info': info, 'meshes': {}}
    for name, solid in {**assembly, **refs}.items():
        mesh = mesh_of(solid)
        data['meshes'][name] = {'vertices': np.round(mesh.vertices, 3).tolist(), 'faces': mesh.faces.tolist()}
    (args.preview_data or ROOT/'preview-mesh.json').write_text(json.dumps(data, separators=(',', ':')))
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
