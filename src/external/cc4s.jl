using YAML

# https://manuals.cc4s.org/user-manual/objects/EigenEnergies.html
# Note: This should be the HF eigenenergies, but here we dump the DFT eigenenergies
function write_eigenenergies(folder::AbstractString,
                             eigenvalues::AbstractVector,
                             εF::Number;
                             force=false)
    @assert length(eigenvalues) == 1
    εk = eigenvalues[1]

    # Note: Eigenenergies need to be ordered in *non-decreasing* order for cc4s !
    @assert maximum(abs, sort(εk) - εk) < 1e-10

    yamlfile     = joinpath(folder, "EigenEnergies.yaml")
    elementsfile = joinpath(folder, "EigenEnergies.elements")
    if !force && (isfile(yamlfile) || isfile(elementsfile))
        error("Generated files $yamlfile and/or $elementsfile exists.")
    end

    metadata = Dict(
        "fermiEnergy" => εF,
        "energies"    => εk,
    )
    data = Dict(
        "version"    => 100,
        "type"       => "Tensor",
        "scalarType" => "Real64",
        "dimensions" => [Dict("length" => length(εk),
                              "type"   => "State")],
        "elements"   => Dict("type" => "TextFile"),
        "unit"       => 1.0,  # DFTK using Hartree as well
        "metaData"   => metadata,
    )
    open(fp -> YAML.write(fp, data), yamlfile, "w")

    open(elementsfile, "w") do fp
        for ε in εk
            println(fp, ε)
        end
    end

    [yamlfile, elementsfile]
end

# See https://manuals.cc4s.org/user-manual/objects/CoulombVertex.html
function write_coulomb_vector(folder::AbstractString, ΓnmF::AbstractArray{T, 5};
                              force=true) where {T}
    n_kpt   = size(ΓnmF, 1)
    n_bands = size(ΓnmF, 2)
    n_aux_field = size(ΓnmF, 5)
    @assert n_kpt   == size(ΓnmF, 3)
    @assert n_bands == size(ΓnmF, 4)
    @assert n_kpt  == 1  # 1 kpt is hard-coded for now (see write_eigenenergies)

    yamlfile     = joinpath(folder, "CoulombVertex.yaml")
    elementsfile = joinpath(folder, "CoulombVertex.elements")
    if !force && (isfile(yamlfile) || isfile(elementsfile))
        error("Generated files $yamlfile and/or $elementsfile exists.")
    end

    dimensions = [
        Dict("length" => n_aux_field,     "type" => "AuxiliaryField"),
        Dict("length" => n_kpt * n_bands, "type" => "State"),
        Dict("length" => n_kpt * n_bands, "type" => "State"),
    ]
    data = Dict(
        "version"    => 100,
        "type"       => "Tensor",
        "scalarType" => "Complex64",
        "dimensions" => dimensions,
        "elements"   => Dict("type" => "IeeeBinaryFile"),
        "unit"       => 1.0,  # DFTK using Hartree as well
        "metaData"   => Dict("halfGrid" => 0),  # Complex integrals
    )
    open(fp -> YAML.write(fp, data), yamlfile, "w")

    # C++ is row-major, julia is column-major. Therefore we write
    # ΓnmF in a stream using chunks of all field Fs for given (n,m)
    open(elementsfile, "w") do fp
        for n = 1:n_bands, m = 1:n_bands
           binary = convert(Vector{Complex{Cdouble}}, vec(ΓnmF[1,n,1,m,:]))
           write(fp, binary)
        end # n,m
    end

    [yamlfile, elementsfile]
end


"""
Write CC4S input files from an SCF result `scfres` or an output
from `compute_bands` to the `folder` (by default `joinpath(pwd(), "cc4s")`).
This will dump a number of `yaml` and binary/text data files
to this folder. The files written will be returned by the function.
If `force` is true then existing files will be overwritten.

!!! warning "Experimental function"
    This function is experimental, may not work in all cases or
    may be changed incompatibly in the future (including patch version bumps).
"""
function export_cc4s(scfres::NamedTuple,
                     folder::AbstractString=joinpath(pwd(), "cc4s");
                     n_bands=scfres.n_bands_converge,
                     force=false, auxfield_thresh=1e-6, Ecut_ratio=2/3)
    # TODO Check for the scfres to be Hartree-Fock ... otherwise the resulting Calculation
    #      is not necessarily proper Coupled Cluster

    mkpath(folder)

    eigenvalues = map(εk -> εk[1:n_bands], scfres.eigenvalues)
    files_ene = write_eigenenergies(folder, eigenvalues, scfres.εF; force)

    ΓmnG = compute_coulomb_vertex(scfres.basis, scfres.ψ; n_bands)

    # set up a filter
    # this only works for Gamma-only now
    Ecut_reduced = scfres.basis.Ecut * Ecut_ratio
    basis = scfres.basis
    kpt = basis.kpoints[1]
    model = basis.model
    G_mask = [sum(abs2, model.recip_lattice * G) / 2 <= Ecut_reduced
              for G in kpt.G_vectors]
    G_indices = findall(G_mask)
    nG_reduced = length(G_indices)
    nk1, nb1, nk2, nb2, nG = size(ΓmnG)
    ΓmnG_reduced = zeros(eltype(ΓmnG), nk1, nb1, nk2, nb2, nG_reduced)
    ΓmnG_reduced[:,:,:,:,:] = ΓmnG[:,:,:,:,G_indices]

    Γcompress = svdcompress_coulomb_vertex(ΓmnG_reduced; thresh=auxfield_thresh)
    files_coul = write_coulomb_vector(folder, Γcompress; force)

    append!(files_ene, files_coul)
end
