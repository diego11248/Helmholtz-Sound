import gmsh

def crear_malla_aula(filename, h_size, Lx=7.5, Ly=6.0, Lz=2.8):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("Aula_NTC4595")

    # Crear caja 3D
    box = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
    gmsh.model.occ.synchronize()

    # Identificar superficies por su centro de masa para asignar Physical Groups
    superficies = gmsh.model.getEntities(dim=2)
    techo_tags = []
    paredes_piso_tags = []

    for dim, tag in superficies:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(com[2] - Lz) < 1e-4:
            techo_tags.append(tag)
        else:
            paredes_piso_tags.append(tag)

    # Grupos físicos: 10=Aire, 20=Paredes/Piso, 21=Techo
    gmsh.model.addPhysicalGroup(3, [box], tag=10, name="Volumen_Aire")
    gmsh.model.addPhysicalGroup(2, paredes_piso_tags, tag=20, name="Paredes_Piso")
    gmsh.model.addPhysicalGroup(2, techo_tags, tag=21, name="Techo")

    # Tamaño de malla
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_size * 0.85)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_size)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

    gmsh.model.mesh.generate(3)
    gmsh.write(filename)
    
    _, node_coords, _ = gmsh.model.mesh.getNodes()
    _, tet_tags, _ = gmsh.model.mesh.getElements(dim=3)
    n_nodos = len(node_coords) // 3
    n_tets = len(tet_tags[0])
    print(f"Generada {filename:<22} | h={h_size:.3f} m | Nodos={n_nodos:<7} | Tet4={n_tets}")
    gmsh.finalize()

if __name__ == "__main__":
    # 1. Mallas para estudio de convergencia h (Tabla 3)
    crear_malla_aula("malla_M1.msh", h_size=1.20)
    crear_malla_aula("malla_M2.msh", h_size=0.65)
    crear_malla_aula("malla_M3.msh", h_size=0.35)

    # 2. Mallas para simulación forzada del aula (Tabla 5)
    # h = lambda / 10 -> a 250 Hz h=0.137m; a 500 Hz h=0.069m (usa 0.09m si tienes <16GB RAM)
    crear_malla_aula("aula_250Hz.msh", h_size=0.14)
    crear_malla_aula("aula_500Hz.msh", h_size=0.085)