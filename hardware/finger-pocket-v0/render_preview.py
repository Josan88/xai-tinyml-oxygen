"""Render the generated geometry as a technical concept illustration."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT/'preview-mesh.json').read_text())


def add(ax, name, color, alpha=1):
    mesh = data['meshes'][name]
    triangles = np.array(mesh['vertices'])[np.array(mesh['faces'])]
    ax.face_buffer.extend(triangles)
    ax.color_buffer.extend([to_rgba(color, alpha)]*len(triangles))


def finish(ax, closed):
    # A depth buffer renders the actual mesh without painter-sort artifacts.
    resolution = 1100
    tri = np.array(ax.face_buffer)
    colors = np.array(ax.color_buffer)[:,:3]
    az, el = np.deg2rad([215, 27 if closed else 53])
    forward = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0])
    up = np.cross(forward, right)
    projected = tri @ np.stack([right, up, forward], axis=1)
    low = projected[:,:,:2].min(axis=(0,1))
    high = projected[:,:,:2].max(axis=(0,1))
    scale = (resolution-100)/max(high-low)
    projected[:,:,:2] = (projected[:,:,:2]-(high+low)/2)*scale+resolution/2
    projected[:,:,1] = resolution-projected[:,:,1]
    rgb = np.empty((resolution, resolution, 3))
    rgb[:] = to_rgba('#f7f7f4')[:3]
    depth = np.full((resolution, resolution), -np.inf)
    normals = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:,None], 1e-12)
    light = np.array([-0.5,-0.7,1.1]); light /= np.linalg.norm(light)
    shades = 0.52+0.48*np.maximum(0, normals@light)
    for t, color, shade in zip(projected, colors, shades):
        xmin,ymin = np.maximum(0, np.floor(t[:,:2].min(axis=0)).astype(int))
        xmax,ymax = np.minimum(resolution-1, np.ceil(t[:,:2].max(axis=0)).astype(int))
        if xmax < xmin or ymax < ymin:
            continue
        (x0,y0,z0),(x1,y1,z1),(x2,y2,z2) = t
        denom = (y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(denom) < 1e-9:
            continue
        xx,yy = np.meshgrid(np.arange(xmin,xmax+1)+0.5, np.arange(ymin,ymax+1)+0.5)
        a = ((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/denom
        b = ((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/denom
        c = 1-a-b
        zz = a*z0+b*z1+c*z2
        target = depth[ymin:ymax+1,xmin:xmax+1]
        mask = (a >= -1e-8)&(b >= -1e-8)&(c >= -1e-8)&(zz > target)
        target[mask] = zz[mask]
        rgb[ymin:ymax+1,xmin:xmax+1][mask] = color*shade
    ax.imshow(rgb, interpolation='antialiased')


def finger(ax):
    # Schematic finger envelope, not a scan. Pad touches z=12 near x=32.
    xs = np.r_[np.linspace(-30, 32, 20), np.linspace(32.01, 40.99, 20)]
    th = np.linspace(0, 2*np.pi, 48)
    rr = np.where(xs <= 32, 1.0, np.sqrt(np.maximum(0, 1-((xs-32)/9)**2)))
    x = np.tile(xs[:, None], (1, len(th)))
    y = 15+9*rr[:, None]*np.cos(th)
    z = 21+9*rr[:, None]*np.sin(th)
    points = np.stack([x,y,z], axis=-1)
    for i in range(len(xs)-1):
        for j in range(len(th)-1):
            ax.face_buffer.extend([[points[i,j],points[i+1,j+1],points[i+1,j]],
                                   [points[i,j],points[i,j+1],points[i+1,j+1]]])
            ax.color_buffer.extend([to_rgba('#d3a293')]*2)
    for j in range(len(th)-1):
        ax.face_buffer.append([[-30,15,21],points[0,j+1],points[0,j]])
        ax.color_buffer.append(to_rgba('#d3a293'))


p = json.loads((ROOT/'dimensions.json').read_text())
fig = plt.figure(figsize=(15, 8), facecolor='#f7f7f4')
fig.text(0.04, 0.94, 'Covered finger-pocket enclosure', fontsize=23,
         fontweight='medium', color='#202b32')
fig.text(0.04, 0.895, 'CONCEPT V0   /   400-point breadboard   /   107 × 90 × 40.4 mm',
         fontsize=12, color='#4c595f')
axes = [fig.add_axes([0.01, 0.22, 0.48, 0.65]),
        fig.add_axes([0.50, 0.22, 0.48, 0.65])]
for idx, ax in enumerate(axes):
    ax.face_buffer, ax.color_buffer = [], []
    ax.set_facecolor('#f7f7f4')
    add(ax, 'base', '#cbd5d9')
    add(ax, 'finger-deck', '#648d98')
    add(ax, 'breadboard', '#e8e5da')
    add(ax, 'controller', '#376f61')
    add(ax, 'sensor', '#376f61')
    add(ax, 'optics', '#b24440')
    if idx == 0:
        add(ax, 'lid', '#cbd5d9')
        finger(ax)
    finish(ax, idx == 0)
    ax.set_axis_off()
fig.text(0.06, 0.205, 'Finger enters from the side', fontsize=15, fontweight='medium', color='#202b32')
fig.text(0.06, 0.164, 'Opaque roof shades the optical contact.', fontsize=12, color='#4c595f')
fig.text(0.54, 0.205, 'Roof removed', fontsize=15, fontweight='medium', color='#202b32')
fig.text(0.54, 0.164, 'Sensor pocket beside the breadboard; wires below the deck.', fontsize=12, color='#4c595f')
fig.text(0.04, 0.075, 'Provisional hardware sizes. Sensor, USB, wiring height and finger fit require a physical trial.',
         fontsize=11.5, color='#4c595f')
fig.savefig(ROOT/'enclosure-preview.png', dpi=160, facecolor=fig.get_facecolor())
plt.close(fig)
print(ROOT/'enclosure-preview.png')
