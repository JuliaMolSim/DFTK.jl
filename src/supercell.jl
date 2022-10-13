"""
Construct a supercell of size `supercell_size` from a unit cell described by its `lattice`,
`atoms` and their positions.
"""
function create_supercell(lattice, atoms, positions, supercell_size)
    supercell = ase_atoms(lattice, atoms, positions) * supercell_size

    lattice_supercell   = load_lattice(supercell)
    positions_supercell = load_positions(supercell)
    atoms_supercell     = load_atoms(supercell)

    (; lattice=lattice_supercell, atoms=atoms_supercell, positions=positions_supercell)
end
