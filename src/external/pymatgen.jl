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


function pymatgen_bandstructure(basis, band_data, εF, klabels)
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
    pylattice = pymatgen_lattice(basis.model.lattice)
    elec_structure.bandstructure.BandStructureSymmLine(
        kcoords, eigenvals, pylattice.reciprocal_lattice, εF,
        labels_dict=klabels, coords_are_cartesian=true
    )
end


"""
Load a DFTK-compatible lattice object from a supported pymatgen object
"""
function load_lattice(T, pyobj::PyObject)
    mg = pyimport("pymatgen")

    if pyisinstance(pyobj, mg.Structure)
        load_lattice(T, pyobj.lattice)
    elseif pyisinstance(pyobj, mg.Lattice)
        lattice = Matrix{T}(undef, 3, 3)
        for i in 1:3, j in 1:3
            lattice[i, j] = units.Ǎ * get(get(pyobj.matrix, j-1), i-1)
        end
        Mat3{T}(lattice)
    else
        error("load_lattice not implemented for python type $pyobj")
    end
end


# One could probably make this proper at some point and
# make it a part of the main code
function guess_psp_for_element(symbol, functional; cheapest=true)
    fun = cheapest ? first : last
    fun(psp for psp in list_psp() for l in 1:30
          if startswith(psp, "hgh/$(lowercase(functional))/$(lowercase(symbol))-q$l"))
end


"""
Load a DFTK-compatible atoms representation from a supported pymatgen object
"""
function load_atoms(T, pyobj::PyObject; functional="lda", pspmap=Dict())
    mg = pyimport("pymatgen")
    pyisinstance(pyobj, mg.Structure) || error("load_atoms is only implemented for " *
                                               "python type pymatgen.Structure")

    map(unique(pyobj.species)) do spec
        coords = [s.frac_coords for s in pyobj.sites if s.specie == spec]
        psp = nothing
        if spec.number in keys(pspmap)
            psp = pspmap[spec.number]
        elseif functional !== nothing
            psp = guess_psp_for_element(spec.symbol, functional)
            @info("Using autodetermined pseudopotential for $(spec.symbol).", psp)
        end
        ElementPsp(spec.number, psp=load_psp(psp)) => coords
    end
end
