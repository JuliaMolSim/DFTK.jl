using HDF5

# See https://manuals.cc4s.org/user-manual/objects/CoulombVertex.html
function compute_coulomb_vertex(basis, ψ)
    T = Float64
    mpi_nprocs(basis.comm_kpts) > 1 && error("Cannot use mpi")
    if length(basis.kpoints) == 1 && basis.use_symmetries_for_kpoint_reduction
        error("Cannot use symmetries")
        # This requires appropriate insertion of kweights
    end

    error("This version of the function should theoretically be correct, but is untested.")

    # TODO Other cases should work, but never tested and could be buggy
    n_kpt = length(basis.kpoints)
    @assert n_kpt == 1

    n_bands = size(ψ[1], 2)
    n_G  = prod(basis.fft_size)
    ΓmnG = zeros(complex(T), n_bands * n_kpt, n_bands * n_kpt, n_G)
    for (ikn, kptn) in enumerate(basis.kpoints), (n, ψnk) in enumerate(eachcol(ψ[ikn]))
        ψnk_real = ifft(basis, kptn, ψnk)
        for (ikm, kptm) in enumerate(basis.kpoints)
            q = kptn.coordinate - kptm.coordinate
            coeffs = sqrt(compute_poisson_green_coeffs(basis, one(T); q))
            for (m, ψmk) in enumerate(eachcol(ψ[ikm]))
                ψmk_real = ifft(basis, kptm, ψmk)
                mm = (ikm - 1) * n_kpt + m  # Blocks of all bands for each k-point
                nn = (ikn - 1) * n_kpt + n
                ΓmnG[mm, nn, :] = coeffs .* fft(basis, conj(ψmk_real) .* ψnk_real)
            end  # ψmk
        end # kptm
    end  # kptn, ψnk
    ΓmnG
end

function twice_coulomb_energy(ΓmnG, occupation)
    occk = only(occupation)  # TODO This routine fails for n_kpt != 1
    n_bands = length(occk)

    res = zero(Float64)
    for (nk, occnk) in enumerate(occk)
        for (mk, occmk) in enumerate(occk)
            res += real(dot(ΓmnG[nk, nk, :], ΓmnG[mk, mk, :])) * occnk * occmk
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
