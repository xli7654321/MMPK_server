import io
import re
import base64
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from rdkit.Chem.Draw import rdMolDraw2D

def show_mol_svg(smi, sub_smi, score, size=(500, 500), cmap_name='YlOrRd'):
    mol = Chem.MolFromSmiles(smi)
    sub_clean = re.sub(r'\(\[\d+\*\]\)', '', sub_smi)
    sub_clean = re.sub(r'\[\d+\*\]', '', sub_clean)
    frag = Chem.MolFromSmiles(sub_clean)

    matches = mol.GetSubstructMatches(frag)
    if not matches:
        return None

    highlight_atoms = list(matches[0])
    
    vmin, vmax = 0, 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    
    highlight_colors = {idx: cmap(norm(score))[:3] for idx in highlight_atoms}
    
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)  # (width, height)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.bondLineWidth = 3
    opts.highlightBondWidthMultiplier = 6
    opts.highlightRadius = 0.35
    opts.padding = 0.05
    opts.fixedFontSize = 20
    opts.rotate = 0
    opts.addAtomIndices = False
    # opts.annotationFontScale = 0.6
    opts.fontFile = 'static/ProximaSoft-Medium.ttf'
    
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors
    )
    drawer.FinishDrawing()
    mol_svg = drawer.GetDrawingText()
    
    return mol_svg

def show_cbar_svg(cmap_name='YlOrRd', figsize=(0.3, 6)):
    vmin, vmax = 0, 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots(figsize=figsize)
    cbar = fig.colorbar(sm, cax=ax)
    cbar.set_ticks(np.arange(vmin, vmax + 0.1, 0.2))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.tick_params(labelsize=18)
    cbar.outline.set_visible(False)

    out = io.StringIO()
    fig.savefig(out, format='svg', bbox_inches='tight')
    plt.close(fig)
    cbar_svg = out.getvalue()
    
    return cbar_svg

def show_sub_svg(sub_smi, size=(500, 500)):
    mol = Chem.MolFromSmiles(sub_smi)
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)  # (width, height)
    
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.bondLineWidth = 3
    opts.fixedFontSize = 25
    opts.rotate = 0
    opts.addAtomIndices = False
    opts.fontFile = 'static/ProximaSoft-Medium.ttf'
    
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)  # drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    
    return svg

def svg_to_data_uri(svg: str) -> str:
    b64 = base64.b64encode(svg.encode('utf-8')).decode()
    return f"data:image/svg+xml;base64,{b64}"

if __name__ == '__main__':
    pass