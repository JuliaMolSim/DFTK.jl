# Implementation of the show function for Model, Kpoints, PlaneWaveBasis
# to avoid clutter in their respective files. For "shorter" files
# (e.g. elements.jl, Energies.jl) the preference is still to keep
# show in the same file to avoid forgetting to add a print statement
# for an added field.

function Base.show(io::IO, model::Model)
    nD = model.n_dim == 3 ? "" : "$(model.n_dim)D, "
    print(io, "Model(", model.model_name, ", ", nD,
          "spin_polarization = ", model.spin_polarization, ")")
end

function Base.show(io::IO, ::MIME"text/plain", model::Model)
    println(io, "Model(", model.model_name, ", $(model.n_dim)D):")
    for i = 1:3
        header = i==1 ? "lattice (in Bohr)" : ""
        showfieldln(io, header, (@sprintf "[%-10.6g, %-10.6g, %-10.6g]" model.lattice[i, :]...))
    end
    showfieldln(io, "unit cell volume", @sprintf "%.5g Bohr³" model.unit_cell_volume)

    if !isempty(model.atoms)
        println(io)
        showfieldln(io, "atoms", chemical_formula(model))
        elements = first.(model.atoms)
        for (i, el) in enumerate(elements)
            header = i==1 ? "atom potentials" : ""
            showfieldln(io, header, el)
        end
    end

    println(io)
    showfieldln(io, "num. electrons",    model.n_electrons)
    showfieldln(io, "spin polarization", model.spin_polarization)
    showfieldln(io, "temperature",       @sprintf "%.5g Ha" model.temperature)
    if model.temperature > 0
        showfieldln(io, "smearing",      model.smearing)
    end

    println(io)
    for (i, term) in enumerate(model.term_types)
        header = i==1 ? "terms" : ""
        showfield(io, header, term)
        i < length(model.term_types) && println(io)
    end
end


# TODO show function for the Kpoint struct


function Base.show(io::IO, basis::PlaneWaveBasis)
    print(io, "PlaneWaveBasis(model = ", basis.model, ", Ecut = ", basis.Ecut, " Ha")
    if !isnothing(basis.kgrid)
        print(io, ", kgrid = ", basis.kgrid)
        if !isnothing(basis.kshift) && !iszero(basis.kshift)
            print(io, ", kshift = ", basis.kshift)
        end
    else
        print(io, ", num. irred. kpoints = ", length(basis.kcoords_global))
    end
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", basis::PlaneWaveBasis)
    println(io, "PlaneWaveBasis discretization:")
    showfieldln(io, "Ecut",     basis.Ecut, " Ha")
    showfieldln(io, "fft_size", basis.fft_size)
    if !basis.variational
        showfieldln(io, "variational", "false")
    end
    showfieldln(io, "kgrid type", "Monkhorst-Pack")
    showfieldln(io, "kgrid",    basis.kgrid)
    if !iszero(basis.kshift)
        showfieldln(io, "kshift",   basis.kshift)
    end
    showfieldln(io, "num. irred. kpoints", length(basis.kcoords_global))

    println(io)
    modelstr = sprint(show, "text/plain", basis.model)
    indent = " " ^ SHOWINDENTION
    print(io, indent, "Discretized " * replace(modelstr, "\n" => "\n" * indent))
end


