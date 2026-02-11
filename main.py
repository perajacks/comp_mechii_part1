#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The SEMFE Heat Transfer Solver
Computational Mechanics

Main Script
"""
import numpy as np
from PreProcessor import read_input_file
from Solver import assemble_global, apply_convection, apply_dirichlet
from Solver import apply_heat_flux, solve_system
from PostProcessor import plot_mesh, plot_mesh_interactive, plot_temperature_field
from PostProcessor import export_temperature_csv
from Solver import element_stiffness_triangle # vgalto


# Import model info
nodes, elems, materials, k, bcs = read_input_file('Ex1.semfe')

# Check Mesh Quality
plot_mesh_interactive(nodes, elems, show=True, filename='interactive_mesh_chimney.html')


nnodes = nodes.shape[0]
nelems = elems.shape[0]


for e in range(nelems):
     conn = elems[e]
     coords = nodes[conn, :2]  # take x,y only
     Ke = element_stiffness_triangle(coords, k=k)

# Assemble global
K = assemble_global(nodes, elems, k=k)
fmod = np.zeros(nodes.shape[0], dtype=float)

# Apply BCs
bc_nodes = [node for node, val in bcs['temperature']]
bc_values = [val for node, val in bcs['temperature']]
heat_flux_bcs = bcs['heat_flux']
conv_bcs = bcs['convection']
fmod       = apply_heat_flux(fmod, nodes, elems,heat_flux_bcs )
Kmod, fmod = apply_convection(K, fmod, nodes, elems, conv_bcs)

Kmod, fmod = apply_dirichlet(K,fmod , bc_nodes, bc_values)






#
u = solve_system(Kmod, fmod)

# Call it in main
plot_temperature_field(nodes, elems, u, filename='temperature_field.png')
plot_mesh(nodes, elems)
export_temperature_csv(nodes, u)




