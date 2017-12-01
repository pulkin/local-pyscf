import json

from pyscf import gto
from dfttools import types, presentation
import numpy
import numericalunits
import svgwrite

from dchf import DCHF


def load_pyscf_model(name, return_data=False, **kwargs):
    default = dict(
        basis='cc-pvdz',
        verbose=0,
    )
    default.update(kwargs)
    with open("models.json", 'r') as f:
        data = json.load(f)

    default["atom"] = data[name]["pyscf-string"]
    if return_data:
        return gto.M(**default), data[name]
    else:
        return gto.M(**default)


def load_pyscf_cluster_model(name, isolated_cluster=False, **kwargs):
    kwargs["return_data"] = True
    m, data = load_pyscf_model(name, **kwargs)
    hf = DCHF(m)
    for d in data["domains"]:
        tot = d["core"] if isolated_cluster else d["core"]+d["buffer"]
        hf.add_domain(tot, core=d["core"])
    return hf


def draw_cluster_model(name, width=400, height=400, **kwargs):
    kwargs["return_data"] = True
    m, data = load_pyscf_model(name, **kwargs)
    coordinates = numpy.array(list(i[1] for i in m._atom))
    mn = coordinates.min(axis=0)
    mx = coordinates.max(axis=0)
    values = list(i[0] for i in m._atom)
    basis = types.Basis(numpy.diag(numpy.maximum(mx-mn, 1))*numericalunits.aBohr)
    cell = types.UnitCell(basis, (coordinates - mn[numpy.newaxis, :])*numericalunits.aBohr, values, c_basis="cartesian")
    # Draw
    d = svgwrite.Drawing(name + "-domains.svg", size=(width, height))
    if "domains" in data:
        n = len(data["domains"])
        _max_rows = 4
        _d_h = min(_max_rows, n)
        _d_w = n // _max_rows + ((n % _max_rows) > 0)
        _d_ratio = 1.0*_d_w/_d_h
        _d_height = height / _d_h
        _d_width = width / (1.0 + _d_ratio) / _d_h
        _m_height = height
        _m_width = width / (1.0 + _d_ratio)
        _domain_color = (55, 55, 255)
        _buffer_color = (165, 165, 255)
        _env_color = (255, 255, 255)
        presentation.svgwrite_unit_cell(cell, d, size=(_m_width, _m_height), show_numbers=True)
        for i, domain in enumerate(data["domains"]):
            x = i // _d_h
            y = i % _d_h
            presentation.svgwrite_unit_cell(
                cell,
                d,
                size=(_d_width, _d_height),
                insert=(_m_width + _d_width*x, _d_height*y),
                show_legend=False,
                hook_atomic_color=lambda atom, color: _domain_color if atom in domain["core"] else _buffer_color if atom in domain["buffer"] else _env_color,
            )
    else:
        presentation.svgwrite_unit_cell(cell, d, size=(width, height), show_numbers=True)
    d.save()
