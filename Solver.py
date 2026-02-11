#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The SEMFE Heat Transfer Solver
Computational Mechanics

Solver Script
"""

# --------------------------
# File: fem.py
# --------------------------
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np



def element_stiffness_triangle(node_coords, k=1.0):
    """
    Linear triangular element stiffness for steady-state conduction (Poisson equation)
    node_coords: (3,2) or (3,3) array of node coordinates
    returns 3x3 element stiffness matrix
    """
    x1, y1 = node_coords[0, 0], node_coords[0, 1]
    x2, y2 = node_coords[1, 0], node_coords[1, 1]
    x3, y3 = node_coords[2, 0], node_coords[2, 1]
    
    
    J = np.array([[x2-x1,x3-x1],[y2-y1,y3-y1]])
    det_J = np.linalg.det(J)
    abs_det_J = np.absolute(det_J)
    area = 0.5*abs_det_J# element area

     # Shape function derivatives (constant over element)
    B = (1/(np.linalg.det(J))) * np.array([[(y2-y3),(y3-y1)],[(x3-x2),(x1-x3)]]) @ np.array([[1,0,-1],[0,1,-1]])
     
    Ke = k * area * (B.T @ B)
    return Ke


def assemble_global(nodes, elems, k=2.5): 
    """
    Assemble global stiffness matrix for triangular mesh
    nodes: Nx2 or Nx3 array
    elems: Mx3 array of node indices (0-based)
    k: thermal conductivity
    returns: K (sparse CSR matrix)
    """
    nnodes = nodes.shape[0]
    nelems = elems.shape[0]
    rows = []
    cols = []
    data = []
   
    for e in range(nelems):
        conn = elems[e]
        coords = nodes[conn, :2]  # take x,y only
       
        Ke = element_stiffness_triangle(coords, k=k)
       
        for i_local, i_global in enumerate(conn):
            
            for j_local, j_global in enumerate(conn):
                # Assemble global Matrix K
                rows.append(i_global) 
                cols.append(j_global) 
                data.append(Ke[i_local, j_local]) 

    K = sp.coo_matrix((data, (rows, cols)), shape=(nnodes, nnodes)).tocsr() 
    
    return K.toarray()



def apply_dirichlet(K, f, bc_nodes, bc_values):
    """
    Apply Dirichlet boundary conditions to the global matrix
    bc_nodes: array of node indices
    bc_values: array of prescribed values
    """

    f = f.copy()
    for node, val in zip(np.atleast_1d(bc_nodes), np.atleast_1d(bc_values)):
        #modify K and f accordingly    
       K[node, :] = 0
       K[:, node] = 0
       K[node, node] = 1
       f[node] = val
    return K, f

def apply_heat_flux(f, nodes, elems, heat_flux_bcs):
    """
    Apply Neumann (heat flux) BCs to load vector.
    Each BC: (elem_id, edge_id, q)
    """
    fmod = f.copy()

    for elem_id, edge_id, q in heat_flux_bcs:
     
    
        conn = elems[int(elem_id)]       
        coords = nodes[conn, :2]      
    
    
        edge_nodes = {
            1: [0, 1],
            2: [1, 2],
            3: [2, 0]
        }[int(edge_id)]

        n1, n2 = edge_nodes
        x1, y1 = coords[n1]
        x2, y2 = coords[n2]
    
        L = np.hypot(x2 - x1, y2 - y1)
      
        fe = (q * L / 2.0) * np.array([1.0, 1.0])

        fmod[conn[edge_nodes]] += fe
    
    return fmod


def apply_convection(K, f, nodes, elems, conv_bcs):
    """
    Apply Robin (convection) BCs to load vector & matrix K.
    Each BC: (elem_id, edge_id, h, Tinf)
    """
    Kmod = K.copy()
    fmod = f.copy()

    for elem_id, edge_id, h, Tinf in conv_bcs:
        
        conn = elems[int(elem_id)]       
        coords = nodes[conn, :2]
        edge_nodes = {
            1: [0,1],
            2: [1,2],
            3: [2,0]
        }[int(edge_id)]
        n1, n2 = edge_nodes
        x1, y1 = coords[n1]
        x2, y2 = coords[n2]
        L = np.hypot(x2 - x1, y2 - y1)
        #Modify K and F accordingly
        
        K = (h * L / 6.0) * np.array([[2.0, 1.0],
                                          [1.0, 2.0]])
        f = (h * Tinf * L / 2.0) * np.array([1.0, 1.0])
        
        fmod[conn[edge_nodes]] += f
        
        global_nodes = conn[edge_nodes]      
        gn1, gn2 = global_nodes              

        Kmod[gn1, gn1] += K[0, 0]
        Kmod[gn1, gn2] += K[0, 1]
        Kmod[gn2, gn1] += K[1, 0]
        Kmod[gn2, gn2] += K[1, 1]
        
    return Kmod, fmod


def solve_system(K, f):
    """Solve the linear system Ku=f"""
    u = spla.spsolve(K, f)
    return u
