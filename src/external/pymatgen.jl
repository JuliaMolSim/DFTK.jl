# Routines for interaction with pymatgen, e.g. converting to
# its structures from the DFTK equivalents

function pymatgen_lattice(lattice::AbstractArray)
    # Notice: Pymatgen uses rows as lattice vectors, so we unpeel
    # our lattice column by column. The default unit in pymatgen is Ǎngström
    mg = pyimport("pymatgen")
    bohr_to_A = 1 / units.Ǎ
    mg.Lattice([Array(bohr_to_A .* lattice[:, i]) for i in 1:3])
end
pymatgen_lattice(model::Model) = pymatgen_lattice(model.lattice)


function pymatgen_structure(model_or_lattice, atoms)
    mg = pyimport("pymatgen")
    pylattice = pymatgen_lattice(model_or_lattice)

    n_species = sum(length(pos) for (spec, pos) in atoms)
    pyspecies = Vector{Int}(undef, n_species)
    pypositions = Array{Vector{Float64}}(undef, n_species)
    ispec = 1
    for (spec, pos) in atoms
        for coord in pos
            pyspecies[ispec] = spec.Z
            pypositions[ispec] = Vector{Float64}(coord)
            ispec = ispec + 1
        end
    end
    @assert ispec == n_species + 1

    mg.Structure(pylattice, pyspecies, pypositions)
end
pymatgen_structure(model::Model) = pymatgen_structure(model, model.atoms)


function pymatgen_bandstructure(basis, λ, εF, klabels)
    Spin = pyimport("pymatgen.electronic_structure.core").Spin
    bandstructure = pyimport("pymatgen.electronic_structure.bandstructure")

    @assert basis.model.spin_polarization in (:none, :spinless, :collinear)
    n_spin   = basis.model.n_spin_components
    n_kcoord = div(length(basis.kpoints), n_spin)

    mg_spins = (Spin.up, Spin.down)
    mg_eigenvals = Dict()
    for σ in 1:n_spin
        eigenvals = Matrix{eltype(λ[1])}(undef, length(λ[1]), n_kcoord)
        for (ito, ik) in enumerate(krange_spin(basis, σ))
            eigenvals[:, ito] = λ[ik]
        end
        mg_eigenvals[mg_spins[σ]] = eigenvals
    end

    # Convert coordinates to Cartesian
    kcoords = [basis.kpoints[ik].coordinate_cart for ik in 1:n_kcoord]
    klabels = Dict(lal => basis.model.recip_lattice * vec for (lal, vec) in pairs(klabels))
    pylattice = pymatgen_lattice(basis.model.lattice)
    bandstructure.BandStructureSymmLine(
        kcoords, mg_eigenvals, pylattice.reciprocal_lattice, εF,
        labels_dict=klabels, coords_are_cartesian=true  # Issues if this is false!
    )
end


function load_lattice_pymatgen(T, pyobj::PyObject)
    mg = pyimport("pymatgen")

    if pyisinstance(pyobj, mg.Structure)
        load_lattice_pymatgen(T, pyobj.lattice)
    elseif pyisinstance(pyobj, mg.Lattice)
        lattice = Matrix{T}(undef, 3, 3)
        for i in 1:3, j in 1:3
            lattice[i, j] = units.Ǎ * get(get(pyobj.matrix, j-1), i-1)
        end
        Mat3{T}(lattice)
    else
        error("load_lattice_pymatgen not implemented for python type $pyobj")
    end
end


"""
Load a DFTK-compatible atoms representation from a supported pymatgen object.
All atoms are using a Coulomb model.
"""
function load_atoms_pymatgen(T, pyobj::PyObject)
    @assert pyisinstance(pyobj, pyimport("pymatgen").Structure)
    map(unique(pyobj.species)) do spec
        coords = [s.frac_coords for s in pyobj.sites if s.specie == spec]
        ElementCoulomb(spec.number) => coords
    end
end
