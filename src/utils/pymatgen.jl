# Routines for interaction with pymatgen, e.g. converting to
# its structures from the DFTK equivalents

function pymatgen_lattice(model)
    # Notice: Pymatgen uses rows as lattice vectors, so we unpeel
    # our lattice column by column.
    mg = pyimport("pymatgen")
    AtoBohr = pyimport("pymatgen.core.units").ang_to_bohr  # Ǎngström to Bohr
    mg.Lattice([Array(AtoBohr .* model.lattice[:, i]) for i in 1:3])
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
    @assert basis.model.spin_polarisation in (:none, :spinless)

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
