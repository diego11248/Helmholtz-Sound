"""
Motor de Elementos Finitos 3D (Tet4 + Tri3) para la ecuación de Helmholtz armónica.
Autores: Diego Gutierrez, Sebastian Flórez
Asignatura: Métodos Numéricos para EDPS - Universidad EAFIT
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import meshio


class Helmholtz3DSolver:
    def __init__(self, mesh_filepath, c_sound=343.0, rho_air=1.225):
        self.filepath = mesh_filepath
        self.c = c_sound
        self.rho = rho_air
        self.puntos = None
        self.celdas_tet4 = None
        self.caras_tri3 = None
        self.caras_techo = None
        self.caras_paredes = None
        self.n_nodos = 0
        self.K_global = None
        self.M_global = None
        self.C_global = None
        self.load_mesh()

    def load_mesh(self):
        """Lee TODOS los bloques Tet4 y Tri3 (separando techo z=z_max de paredes/piso)."""
        malla = meshio.read(self.filepath)
        self.puntos = malla.points
        self.n_nodos = len(self.puntos)

        tets_list = []
        tris_list = []

        for cell_block in malla.cells:
            if cell_block.type == "tetra":
                tets_list.append(cell_block.data)
            elif cell_block.type == "triangle":
                tris_list.append(cell_block.data)

        if not tets_list:
            raise ValueError(f"No se encontraron elementos Tet4 en {self.filepath}.")

        self.celdas_tet4 = np.vstack(tets_list)
        self.caras_tri3 = np.vstack(tris_list) if tris_list else None

        # Clasificar triángulos de frontera: Techo (z == z_max) vs Paredes/Piso
        if self.caras_tri3 is not None:
            z_max = np.max(self.puntos[:, 2])
            z_centroids = np.mean(self.puntos[self.caras_tri3, 2], axis=1)
            mask_techo = np.abs(z_centroids - z_max) < 1e-4
            self.caras_techo = self.caras_tri3[mask_techo]
            self.caras_paredes = self.caras_tri3[~mask_techo]

    def ensamblar_volumen(self):
        """Ensambla las matrices globales dispersas K y M mediante vectorización NumPy."""
        coords = self.puntos[self.celdas_tet4]  # (N_tet, 4, 3)
        n_elem = len(self.celdas_tet4)

        A_mat = np.ones((n_elem, 4, 4))
        A_mat[:, :, 1:] = coords
        vols = np.abs(np.linalg.det(A_mat)) / 6.0

        if np.any(vols <= 1e-14):
            raise ValueError("Se detectaron tetraedros con volumen nulo o invertido.")

        inv_A = np.linalg.inv(A_mat)
        G = inv_A[:, 1:, :]  # Gradientes constantes (N_tet, 3, 4)

        # Ke = vol * G^T @ G -> (N_tet, 4, 4)
        Ke_all = vols[:, None, None] * np.einsum('eij,eik->ejk', G, G)
        M_ref = (1.0 / 20.0) * (np.ones((4, 4)) + np.eye(4))
        Me_all = vols[:, None, None] * M_ref[None, :, :]

        I_inds = np.repeat(self.celdas_tet4, 4, axis=1).ravel()
        J_inds = np.tile(self.celdas_tet4, (1, 4)).ravel()

        self.K_global = sp.coo_matrix(
            (Ke_all.ravel(), (I_inds, J_inds)), shape=(self.n_nodos, self.n_nodos)
        ).tocsc()
        self.M_global = sp.coo_matrix(
            (Me_all.ravel(), (I_inds, J_inds)), shape=(self.n_nodos, self.n_nodos)
        ).tocsc()

    def _matriz_robin_superficie(self, caras, alpha):
        """Calcula la contribución dispersa C para un subconjunto de triángulos Tri3."""
        if caras is None or len(caras) == 0 or alpha <= 0.0:
            return sp.csc_matrix((self.n_nodos, self.n_nodos), dtype=float)

        beta = (1.0 - np.sqrt(1.0 - alpha)) / (1.0 + np.sqrt(1.0 - alpha))
        pts = self.puntos[caras]  # (N_tri, 3, 3)
        v1 = pts[:, 1, :] - pts[:, 0, :]
        v2 = pts[:, 2, :] - pts[:, 0, :]
        areas = 0.5 * np.linalg.norm(np.cross(v1, v2), axis=1)

        C_ref = (1.0 / 12.0) * (np.ones((3, 3)) + np.eye(3))
        Ce_all = (beta * areas)[:, None, None] * C_ref[None, :, :]

        I_inds = np.repeat(caras, 3, axis=1).ravel()
        J_inds = np.tile(caras, (1, 3)).ravel()
        return sp.coo_matrix(
            (Ce_all.ravel(), (I_inds, J_inds)), shape=(self.n_nodos, self.n_nodos)
        ).tocsc()

    def ensamblar_frontera_impedancia(self, alpha_paredes=0.03, alpha_techo=0.03):
        """Ensambla la matriz global de amortiguamiento Robin C_global."""
        C_paredes = self._matriz_robin_superficie(self.caras_paredes, alpha_paredes)
        C_techo = self._matriz_robin_superficie(self.caras_techo, alpha_techo)
        self.C_global = C_paredes + C_techo

    def resolver_autovalores(self, num_modos=8, sigma=-0.01):
        """Resuelve el problema espectral homogéneo K p = k^2 M p mediante shift-invert."""
        if self.K_global is None or self.M_global is None:
            self.ensamblar_volumen()

        autovalores, autovectores = spla.eigs(
            self.K_global, k=num_modos, M=self.M_global, sigma=sigma, which='LM'
        )
        lambdas = np.clip(np.real(autovalores), 0.0, None)
        idx_sort = np.argsort(lambdas)
        frecuencias = self.c * np.sqrt(lambdas[idx_sort]) / (2.0 * np.pi)
        return frecuencias, autovectores[:, idx_sort]

    def resolver_sistema_forzado(self, freq_hz, pos_fuente=(3.75, 1.0, 1.5), Q0=1e-4,
                                 alpha_paredes=0.03, alpha_techo=0.03):
        """Resuelve el sistema dinámico complejo (K - k^2 M + i k C) p = b."""
        omega = 2.0 * np.pi * freq_hz
        k = omega / self.c

        if self.K_global is None or self.M_global is None:
            self.ensamblar_volumen()

        self.ensamblar_frontera_impedancia(alpha_paredes=alpha_paredes, alpha_techo=alpha_techo)

        A_sys = self.K_global - (k**2) * self.M_global + 1j * k * self.C_global

        # Ubicar nodo más cercano a la fuente monopolar y aplicar b_m = i * omega * rho * Q0
        dists = np.linalg.norm(self.puntos - np.array(pos_fuente), axis=1)
        nodo_fuente = np.argmin(dists)

        b = np.zeros(self.n_nodos, dtype=complex)
        b[nodo_fuente] = 1j * omega * self.rho * Q0

        p_complex = spla.spsolve(A_sys, b)

        p_ref = 2e-5  # Presión de referencia en aire (20 uPa)
        spl_db = 20.0 * np.log10(np.maximum(np.abs(p_complex), 1e-12) / p_ref)
        return p_complex, spl_db

    def metricas_plano_escucha(self, spl_db, z_plano=1.20, tol_z=0.08):
        """Calcula SPL mínimo, máximo, promedio y desviación estándar en z ≈ z_plano."""
        mask_plano = np.abs(self.puntos[:, 2] - z_plano) <= tol_z
        if not np.any(mask_plano):
            raise ValueError(f"No hay nodos dentro de z = {z_plano} ± {tol_z} m.")
        spl_plano = spl_db[mask_plano]
        return (
            float(np.min(spl_plano)),
            float(np.max(spl_plano)),
            float(np.mean(spl_plano)),
            float(np.std(spl_plano)),
        )

    def exportar_vtk(self, filename, campos_dict):
        """Exporta los resultados nodales en formato VTK para visualizar en ParaView."""
        mesh_out = meshio.Mesh(
            points=self.puntos,
            cells=[("tetra", self.celdas_tet4)],
            point_data=campos_dict,
        )
        mesh_out.write(filename)