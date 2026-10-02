import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import meshio

def matrices_elementales_tet4(coords_tet):
    """Calcula las matrices de rigidez Ke y masa Me para un tetraedro Tet4."""
    A_mat = np.ones((4, 4))
    A_mat[:, 1:] = coords_tet
    vol = abs(np.linalg.det(A_mat)) / 6.0  
    
    if vol <= 1e-12:
        raise ValueError("Volumen de elemento cero o extremadamente pequeño.")

    inv_A = np.linalg.inv(A_mat)
    grad_N = inv_A[1:, :]
    
    Ke = vol * (grad_N.T @ grad_N)
    Me = (vol / 20.0) * (np.ones((4, 4)) + np.identity(4))
    return Ke, Me, vol

def cargar_malla_gmsh(filepath):
    malla = meshio.read(filepath)
    puntos = malla.points
    celdas_tet4 = None
    for cell_block in malla.cells:
        if cell_block.type == "tetra":
            celdas_tet4 = cell_block.data
            break
    if celdas_tet4 is None:
        raise ValueError("No se encontraron elementos tetraédricos (Tet4) en el archivo .msh.")
    return puntos, celdas_tet4

def resolver_autovalores_helmholtz(puntos, celdas, c_sound=343.0, num_modos=6):
    n_nodos = len(puntos)
    I_inds, J_inds, K_vals, M_vals = [], [], [], []
    
    for tet in celdas:
        coords = puntos[tet]
        Ke, Me, vol = matrices_elementales_tet4(coords)
        for i in range(4):
            for j in range(4):
                I_inds.append(tet[i])
                J_inds.append(tet[j])
                K_vals.append(Ke[i, j])
                M_vals.append(Me[i, j])
                
    K_global = sp.coo_matrix((K_vals, (I_inds, J_inds)), shape=(n_nodos, n_nodos)).tocsc()
    M_global = sp.coo_matrix((M_vals, (I_inds, J_inds)), shape=(n_nodos, n_nodos)).tocsc()
    
    autovalores, autovectores = spla.eigs(K_global, k=num_modos, M=M_global, sigma=-0.01, which='LM')
    lambdas = np.clip(np.real(autovalores), 0, None)
    frecuencias = c_sound * np.sqrt(lambdas) / (2.0 * np.pi)
    
    return frecuencias, autovectores

if __name__ == "__main__":
    # Malla de validación: caja rígida de 7.5 x 6.0 x 2.8 m
    archivo_msh = "caja-val.msh"
    
    puntos, celdas = cargar_malla_gmsh(archivo_msh)
    print(f"Malla cargada con éxito: {len(puntos)} nodos, {len(celdas)} tetraedros.")
    
    Lx, Ly, Lz = 7.5, 6.0, 2.8  # Dimensiones del aula para 25 estudiantes (45 m²)
    c_sound = 343.0
    num_modos = 8
    
    frec_num, _ = resolver_autovalores_helmholtz(puntos, celdas, c_sound=c_sound, num_modos=num_modos)
    frec_num = np.sort(frec_num)
    
    # Modos analíticos ordenados por frecuencia (no por índice), para emparejarlos con los numéricos
    analiticos = sorted(
        ((c_sound / 2.0) * np.sqrt((nx/Lx)**2 + (ny/Ly)**2 + (nz/Lz)**2), (nx, ny, nz))
        for nx in range(4) for ny in range(4) for nz in range(4)
    )[:num_modos]

    for f_num, (f_ana, (nx, ny, nz)) in zip(frec_num, analiticos):
        error = abs(f_num - f_ana) / f_ana * 100.0 if f_ana > 0 else 0.0
        print(f"Modo ({nx},{ny},{nz}): Analítica = {f_ana:.2f} Hz | Numérica = {f_num:.2f} Hz | Error = {error:.2f}%")