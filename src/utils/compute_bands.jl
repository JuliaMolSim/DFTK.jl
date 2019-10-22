using PyCall
include("pymatgen.jl")

# Functionality for computing band structures, mostly using pymatgen

function determine_high_symmetry_kpath(basis, kline_density, composition...)
    bandstructure = pyimport("pymatgen.symmetry.bandstructure")
    pystructure = pymatgen_structure(basis.model, composition...)
    symm_kpath = bandstructure.HighSymmKpath(pystructure)

    kcoords, labels = symm_kpath.get_kpoints(kline_density, coords_are_cartesian=false)
    kpoints = build_kpoints(basis, kcoords)

    labels_dict = Dict{String, Vector{eltype(kcoords[1])}}()
    for (ik, k) in enumerate(kcoords)
        if length(labels[ik]) > 0
            labels_dict[labels[ik]] = k
        end
    end
    println(symm_kpath.kpath["path"])

    (kpoints=kpoints, klabels=labels_dict, kpath=symm_kpath.kpath["path"][1])
end


function compute_bands(ham::Hamiltonian, kpoints, n_bands; diag=diag_lobpcg())
    prec = PreconditionerKinetic(ham, α=0.5)
    band_data = diag(ham, n_bands + 3; kpoints=kpoints, n_conv_check=n_bands, prec=prec,
                     interpolate_kpoints=false)
    band_data.converged || (@warn "LOBPCG not converged" iterations=eigres.iterations)
    band_data
end
