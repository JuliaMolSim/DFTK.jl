# Routines for interaction with pymatgen, e.g. converting to
# its structures from the DFTK equivalents

function pymatgen_lattice(lattice)
    # Notice: Pymatgen uses rows as lattice vectors, so we unpeel
    # our lattice column by column. The default unit in pymatgen is Ǎngström
    mg = pyimport("pymatgen")
    bohr_to_A = 1 / pyimport("pymatgen.core.units").ang_to_bohr
    mg.Lattice([Array(bohr_to_A .* lattice[:, i]) for i in 1:3])
end


function pymatgen_structure(lattice, composition...)
    mg = pyimport("pymatgen")
    pylattice = pymatgen_lattice(lattice)

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

    # The energy unit in pymatgen is eV
    Ha_to_eV = 1 / pyimport("pymatgen.core.units").eV_to_Ha

    # This assumes no spin polarisation
    @assert basis.model.spin_polarisation in (:none, :spinless)

    kpoints = band_data.kpoints
    n_bands = length(band_data.λ[1])
    eigenvals_spin_up = Matrix{eltype(band_data.λ[1])}(undef, n_bands, length(kpoints))
    for (ik, λs) in enumerate(band_data.λ)
        eigenvals_spin_up[:, ik] = λs * Ha_to_eV
    end
    eigenvals = Dict(elec_structure.core.Spin.up => eigenvals_spin_up)

    kcoords = [kpt.coordinate for kpt in kpoints]
    pylattice = pymatgen_lattice(basis.model.lattice)
    elec_structure.bandstructure.BandStructureSymmLine(
        kcoords, eigenvals, pylattice.reciprocal_lattice, fermi_level * Ha_to_eV,
        labels_dict=klabels, coords_are_cartesian=true
    )
end
