using HDF5

# See https://manuals.cc4s.org/user-manual/objects/CoulombVertex.html
function compute_coulomb_vertex(basis, ψ)
    @assert length(basis.kpoints) == 1

    hartree = only(t for t in basis.terms if t isa TermHartree)
    coeff   = sqrt.(hartree.poisson_green_coeffs)

    kpt = only(basis.kpoints)
    ψk  = only(ψ)

    n_bands = size(ψk, 2)
    n_G   = prod(basis.fft_size)
    ΓmnGk = zeros(ComplexF64, n_bands, n_bands, n_G)

    for (nk, ψnk) in enumerate(eachcol(ψk))
        ψnk_real = ifft(basis, kpt, ψnk)
        for (mk, ψmk) in enumerate(eachcol(ψk))
            ψmk_real = ifft(basis, kpt, ψmk)
            kweight = sqrt(basis.kweights[1])
            ΓmnGk[mk, nk, :] = kweight * coeff .* fft(basis, conj(ψmk_real) .* ψnk_real)
        end
    end

    ΓmnGk
end

function twice_coulomb_energy(ΓmnGk, occupation)
    occk = only(occupation)
    n_bands = length(occk)

    res = zero(Float64)
    for (nk, occnk) in enumerate(occk)
        for (mk, occmk) in enumerate(occk)
            res += real(dot(ΓmnGk[nk, nk, :], ΓmnGk[mk, mk, :])) * occnk * occmk
        end
    end

    res
end



function export_cc4s(datafile::AbstractString, scfres)
    basis = scfres.basis
    ΓmnGk = compute_coulomb_vertex(scfres.basis, scfres.ψ)

    εk = only(scfres.eigenvalues)
    h5open(datafile, "w") do file
        write(file, "Gamma_mnG", ΓmnGk)
        write(file, "eigenvalues", εk)
        write(file, "epsilon_fermi", scfres.εF)
    end
end
