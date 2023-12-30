using HDF5

function export_cc4s(datafile::AbstractString, scfres)
    basis = scfres.basis
    @assert length(basis.kpoints) == 1

    hartree = only(t for t in scfres.basis.terms if t isa TermHartree)
    coeff = sqrt.(hartree.poisson_green_coeffs)

    kpt = only(scfres.basis.kpoints)
    ψk  = only(scfres.ψ)
    εk  = only(scfres.eigenvalues)

    # See https://manuals.cc4s.org/user-manual/objects/CoulombVertex.html
    n_bands = length(εk)
    n_G   = prod(scfres.basis.fft_size)
    ΓmnGk = zeros(ComplexF64, n_bands, n_bands, n_G)
    @showprogress for (nk, ψnk) in enumerate(eachcol(ψk))
        ψnk_real = ifft(scfres.basis, kpt, ψnk)
        for (mk, ψmk) in enumerate(eachcol(ψk))
            ψmk_real = ifft(scfres.basis, kpt, ψmk)
            ΓmnGk[nk, mk, :] = coeff .* sum(ψmk_real .* ψnk_real * basis.dvol)
        end
    end

    h5open(datafile, "w") do file
        write(file, "Gamma_mnG", ΓmnGk)
        write(file, "eigenvalues", εk)
        write(file, "epsilon_fermi", scfres.εF)
    end
end
