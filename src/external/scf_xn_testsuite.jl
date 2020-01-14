# Parser functionality for folders from the https://github.com/NickWoods1/scf-xn-testsuite
# Needs the python package `castep_parse`, available from pypi

using PyCall

struct ScfXnFolder
    folder
    cell  # Parsed cell file as Dict
end


function ScfXnFolder(folder::AbstractString)
    parser = pyimport("castep_parse")
    cellfiles = [f for f in readdir(folder) if splitext(f)[2] == ".cell"]
    if length(cellfiles) != 1
        error("The folder $folder contained none or more than one cell file")
    end
    ScfXnFolder(folder, parser.read_cell_file(joinpath(folder, cellfiles[1])))
end


function load_lattice(T, folder::ScfXnFolder)
    A_to_bohr = pyimport("pymatgen.core.units").ang_to_bohr

    # CASTEP cell files contain lattice vectors in rows
    # and in units of Angström. So the implicit python-julia
    # transpose means that this gets us the lattice vectors
    # in columns:
    lattice = Matrix{T}(undef, 3, 3)
    for i in 1:3
        lattice[:, i] .= folder.cell["supercell"][:, i] * A_to_bohr
    end
    Mat3(lattice)
end


function load_composition(T, folder::ScfXnFolder)
    lattice = load_lattice(T, folder)
    bohr_to_A = 1 / pyimport("pymatgen.core.units").ang_to_bohr
    lattice_A = bohr_to_A * lattice

    composition = Dict{Species, Vector{Vec3{T}}}()
    n_species = length(folder.cell["species"])
    for ispec in 1:n_species
        symbol = get(folder.cell["species"], ispec - 1)
        pspfile = guess_psp_for_element(symbol, "pbe")
        Z = pyimport("pymatgen").Element(symbol).Z
        spec = Species(Z, psp=load_psp(pspfile))

        # CASTEP cell files contain lattice vectors in rows
        # and in units of Angström. So the implicit python-julia
        # transpose means that this gets us the lattice vectors
        # in columns:
        mask_species = findall(isequal(ispec - 1), folder.cell["species_idx"])
        positions_cart = folder.cell["atom_sites"][:, mask_species]
        composition[spec] = [inv(lattice_A) * Vec3{T}(poscol)
                             for poscol in eachcol(positions_cart)]
    end
    pairs(composition)
end


function load_model(T, folder::ScfXnFolder)
    # According to the paper https://doi.org/10.1088/1361-648X/ab31c0,
    # they use a Gaussian smearing with T = 300 K, in AU this is:
    Tsmear = 0.0009500431544769484

    composition = load_composition(T, folder)
    model_dft(Array{T}(load_lattice(folder)), [:gga_x_pbe, :gga_c_pbe],
              composition..., smearing=smearing_gaussian, temperature=Tsmear)
end


function load_basis(T, folder::ScfXnFolder; Ecut=30)
    model = load_model(T, folder)
    composition = load_composition(T, folder)

    # According to the paper https://doi.org/10.1088/1361-648X/ab31c0,
    # they use k-Point spacing 2π * 0.04 Ǎ^{-1}
    spacing = 2π * 0.04 / units.Ǎ
    kgrid_size = kgrid_size_from_minimal_spacing(model.lattice, spacing)
    kcoords, ksymops = bzmesh_ir_wedge(kgrid_size, model.lattice, composition...)
    PlaneWaveBasis(model, Ecut, kcoords, ksymops)
end
