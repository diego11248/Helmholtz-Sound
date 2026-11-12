"""
Script de ejecución de experimentos numéricos para el artículo científico:
1. Validación espectral (Tabla 2) y estudio de convergencia h (Tabla 3).
2. Simulación forzada del aula NTC 4595 con y sin tratamiento acústico (Tabla 5).
"""

import time
import numpy as np
from helmholtz_solver import Helmholtz3DSolver


def experimento_validacion_y_convergencia(Lx=7.5, Ly=6.0, Lz=2.8, c=343.0):
    print("\n" + "=" * 80)
    print(" EXPERIMENTO 1A: VALIDACIÓN ESPECTRAL EN MALLA M1 (TABLA 2)")
    print("=" * 80)

    solver_m1 = Helmholtz3DSolver("malla_M1.msh", c_sound=c)
    num_modos = 8
    frecs_m1, _ = solver_m1.resolver_autovalores(num_modos=num_modos)

    # Frecuencias analíticas exactas (excluyendo modo trivial (0,0,0))
    analiticos = sorted(
        ((c / 2.0) * np.sqrt((nx / Lx) ** 2 + (ny / Ly) ** 2 + (nz / Lz) ** 2), (nx, ny, nz))
        for nx in range(4) for ny in range(4) for nz in range(4)
        if (nx, ny, nz) != (0, 0, 0)
    )[: num_modos - 1]

    print(f"{'Modo (nx,ny,nz)':<18} | {'Analítica (Hz)':<15} | {'Numérica (Hz)':<15} | {'Error (%)':<10}")
    print("-" * 68)
    for f_num, (f_ana, (nx, ny, nz)) in zip(frecs_m1[1:], analiticos):
        err = abs(f_num - f_ana) / f_ana * 100.0
        print(f"({nx},{ny},{nz}){'':<11} | {f_ana:<15.2f} | {f_num:<15.2f} | {err:<10.2f}")

    print("\n" + "=" * 80)
    print(" EXPERIMENTO 1B: ESTUDIO DE CONVERGENCIA h (TABLA 3)")
    print("=" * 80)

    f_ana_100 = (c / 2.0) * (1.0 / Lx)
    f_ana_010 = (c / 2.0) * (1.0 / Ly)

    mallas_conv = [
        ("M1", "malla_M1.msh", 1.20),
        ("M2", "malla_M2.msh", 0.65),
        ("M3", "malla_M3.msh", 0.35),
    ]
    errores_100 = []
    hs = []

    for nombre, archivo, h_val in mallas_conv:
        solver = Helmholtz3DSolver(archivo, c_sound=c)
        frecs, vecs = solver.resolver_autovalores(num_modos=8)

        err_100 = abs(frecs[1] - f_ana_100) / f_ana_100 * 100.0
        err_010 = abs(frecs[2] - f_ana_010) / f_ana_010 * 100.0
        errores_100.append(err_100)
        hs.append(h_val)

        orden_p = "--"
        if len(errores_100) > 1:
            p_val = np.log(errores_100[-2] / errores_100[-1]) / np.log(hs[-2] / hs[-1])
            orden_p = f"{p_val:.2f}"

        print(
            f"{nombre} | h={h_val:.2f}m | Nodos={solver.n_nodos:<6} | Tet4={len(solver.celdas_tet4):<6} | "
            f"Err(1,0,0)={err_100:.2f}% | Err(0,1,0)={err_010:.2f}% | p={orden_p}"
        )

        if nombre == "M3":
            solver.exportar_vtk(
                "modos_propios_M3.vtk",
                {
                    "Modo_100": np.real(vecs[:, 1]),
                    "Modo_010": np.real(vecs[:, 2]),
                    "Modo_110": np.real(vecs[:, 3]),
                },
            )


def experimento_aula_forzada():
    print("\n" + "=" * 80)
    print(" EXPERIMENTO 2: RESPUESTA FORZADA EN PLANO DE ESCUCHA z = 1.20 m (TABLA 5)")
    print("=" * 80)

    casos_aula = [
        (250.0, "aula_250Hz.msh"),
        (500.0, "aula_500Hz.msh"),
    ]

    for freq, msh_file in casos_aula:
        solver_aula = Helmholtz3DSolver(msh_file)

        # Escenario A: Aula rígida sin tratamiento (alpha = 0.03 en todas las superficies)
        t0 = time.time()
        p_A, spl_A = solver_aula.resolver_sistema_forzado(
            freq_hz=freq, pos_fuente=(3.75, 1.0, 1.5), Q0=1e-4,
            alpha_paredes=0.03, alpha_techo=0.03
        )
        t_A = time.time() - t0
        min_A, max_A, prom_A, std_A = solver_aula.metricas_plano_escucha(spl_A, z_plano=1.20)

        # Escenario B: Aula con techo acústico (alpha_techo = 0.65, paredes = 0.15)
        t0 = time.time()
        p_B, spl_B = solver_aula.resolver_sistema_forzado(
            freq_hz=freq, pos_fuente=(3.75, 1.0, 1.5), Q0=1e-4,
            alpha_paredes=0.15, alpha_techo=0.65
        )
        t_B = time.time() - t0
        min_B, max_B, prom_B, std_B = solver_aula.metricas_plano_escucha(spl_B, z_plano=1.20)

        print(f"\n--- Frecuencia: {freq:.0f} Hz (Nodos={solver_aula.n_nodos}, Tet4={len(solver_aula.celdas_tet4)}) ---")
        print(f"Escenario A (Rígido)  | Min={min_A:.1f} dB | Max={max_A:.1f} dB | Prom={prom_A:.1f} dB | Desv={std_A:.2f} dB | t={t_A:.2f}s")
        print(f"Escenario B (Tratado) | Min={min_B:.1f} dB | Max={max_B:.1f} dB | Prom={prom_B:.1f} dB | Desv={std_B:.2f} dB | t={t_B:.2f}s")

        solver_aula.exportar_vtk(
            f"resultados_aula_{int(freq)}Hz.vtk",
            {
                "SPL_Escenario_A_dB": spl_A,
                "SPL_Escenario_B_dB": spl_B,
                "Presion_Real_B": np.real(p_B),
            },
        )


if __name__ == "__main__":
    experimento_validacion_y_convergencia()
    experimento_aula_forzada()