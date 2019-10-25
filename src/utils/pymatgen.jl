# Routines for interaction with pymatgen, e.g. converting to
# its structures from the DFTK equivalents

function pymatgen_lattice(model)
    # Notice: Pymatgen uses rows as lattice vectors, but due to some
    # quirks with the conversion between Array and ArrayWithUnit,
    # the usual transpose of the Array between column-major (Julia)
    # and row-major (python) is not done, such that our data layout
    # (lattice vectors in columns, column-major) and the one expected
    # by pymatgen (vectors in rows, but row-major) agree on the memory
    # level and *no transpose* is needed here.
    mg = pyimport("pymatgen")
    mg.Lattice(mg.ArrayWithUnit(Array(model.lattice), "bohr"))
end


function pymatgen_structure(model, composition...)
    mg = pyimport("pymatgen")
    pylattice = pymatgen_lattice(model)

    n_species = sum(length(pos) for (spec, pos) in composition)
    pyspecies = Vector{Int}(undef, n_species)
    pypositions = Array{Vector{Float64}}(undef, n_species)
    ispec = 1
    for (spec, pos) in composition
        for coord in pos
            pyspecies[ispec] = spec.Znuc
            pypositions[ispec] = Vector{Float64}(coord)
            ispec = ispec + 1
        end
    end
    @assert ispec == n_species + 1

    mg.Structure(pylattice, pyspecies, pypositions)
end


function pymatgen_bandstructure(basis, band_data, klabels=Dict{String, Vector{Float64}}(); fermi_level=0.0)
    mg = pyimport("pymatgen")
    elec_structure = pyimport("pymatgen.electronic_structure")

    # This assumes no spin polarisation
    @assert basis.model.spin_polarisation == :none

    kpoints = band_data.kpoints
    n_bands = length(band_data.λ[1])
    eigenvals_spin_up = Matrix{eltype(band_data.λ[1])}(undef, n_bands, length(kpoints))
    for (ik, λs) in enumerate(band_data.λ)
        eigenvals_spin_up[:, ik] = λs
    end
    eigenvals = Dict(elec_structure.core.Spin.up => eigenvals_spin_up)

    kcoords = [kpt.coordinate for kpt in kpoints]
    pylattice = pymatgen_lattice(basis.model)
    elec_structure.bandstructure.BandStructureSymmLine(
        kcoords, eigenvals, pylattice.reciprocal_lattice, fermi_level,
        labels_dict=klabels, coords_are_cartesian=true
    )
end
