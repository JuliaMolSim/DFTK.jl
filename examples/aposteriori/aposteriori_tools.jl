import KrylovKit: ArnoldiIterator, Orthogonalizer, OrthonormalBasis, KrylovDefaults, orthogonalize!
using KrylovKit

############################# ERROR AND RESIDUAL ###############################

# compute the error on the orbitals by aligning the eigenvectors
# this is done by solving min |ϕ - ψ*U| for U unitary matrix of size NxN
# whose solution is U = M(M^*M)^-1/2 where M = ψ^*ϕ
function compute_error(basis, ϕ, ψ)

    # necessary quantites
    Nk = length(basis.kpoints)

    # compute error
    err = similar(ϕ)
    for ik = 1:Nk
        ϕk = ϕ[ik]
        ψk = ψ[ik]
        # compute overlap matrix
        M = ψk'ϕk
        U = M*(M'M)^(-1/2)
        err[ik] = ϕk - ψk*U
    end
    err
end

############################## TANGENT SPACE TOOLS #############################

## projection on frequencies higher than Ecut
function keep_HF(δϕ, basis, Ecut)

    Nk = length(basis.kpoints)

    δφ = deepcopy(δϕ)

    for ik in 1:Nk
        kpt = basis.kpoints[ik]
        G_vec = G_vectors(kpt)
        recip_lat = kpt.model.recip_lattice
        N = size(δφ[ik], 2)

        for i in 1:N
            for g in 1:length(δφ[ik][:,i])
                if sum(abs2, recip_lat * (G_vec[g] + kpt.coordinate)) <= 2*Ecut
                    δφ[ik][g,i] = 0
                end
            end
        end
    end

    δφ
end

## projection on frequencies smaller than Ecut
function keep_LF(δϕ, basis, Ecut)

    Nk = length(basis.kpoints)

    δφ = deepcopy(δϕ)

    for ik in 1:Nk
        kpt = basis.kpoints[ik]
        G_vec = G_vectors(kpt)
        recip_lat = kpt.model.recip_lattice
        N = size(δφ[ik], 2)

        for i in 1:N
            for g in 1:length(δφ[ik][:,i])
                if sum(abs2, recip_lat * (G_vec[g] + kpt.coordinate)) > 2*Ecut
                    δφ[ik][g,i] = 0
                end
            end
        end
    end

    δφ
end

############################## CHANGES OF NORMS ################################

## T = -1/2 Δ + t

function apply_inv_T(Pks, δφ)
    Nk = length(Pks)

    ϕ = []

    for ik = 1:Nk
        ϕk = similar(δφ[ik])
        N = size(δφ[ik], 2)
        Pk = Pks[ik]
        for i = 1:N
            ϕk[:,i] .= 1 ./ (Pk.mean_kin[i] .+ Pk.kin) .* δφ[ik][:,i]
        end
        append!(ϕ, [ϕk])
    end
    ϕ
end

function apply_inv_sqrt_T(Pks, δφ)
    Nk = length(Pks)

    ϕ = []

    for ik = 1:Nk
        ϕk = similar(δφ[ik])
        N = size(δφ[ik], 2)
        Pk = Pks[ik]
        for i = 1:N
            ϕk[:,i] .= 1 ./ sqrt.(Pk.mean_kin[i] .+ Pk.kin) .* δφ[ik][:,i]
        end
        append!(ϕ, [ϕk])
    end
    ϕ
end

function apply_sqrt_T(Pks, δφ)
    Nk = length(Pks)

    ϕ = []

    for ik = 1:Nk
        ϕk = similar(δφ[ik])
        N = size(δφ[ik], 2)
        Pk = Pks[ik]
        for i = 1:N
            ϕk[:,i] .= sqrt.(Pk.mean_kin[i] .+ Pk.kin) .* δφ[ik][:,i]
        end
        append!(ϕ, [ϕk])
    end
    ϕ
end

function apply_M(φ, Pks, δφ)
    δφ = proj_tangent(δφ, φ)
    δφ = apply_sqrt_T(Pks, δφ)
    δφ = proj_tangent(δφ, φ)
    δφ = apply_sqrt_T(Pks, δφ)
    δφ = proj_tangent(δφ, φ)
end

function apply_sqrt_M(φ, Pks, δφ)
    δφ = proj_tangent(δφ, φ)
    δφ = apply_sqrt_T(Pks, δφ)
    δφ = proj_tangent(δφ, φ)
end

function apply_inv_sqrt_M(basis, φ, Pks, res)
    Nk = length(Pks)

    pack(φ) = pack_arrays(basis, φ)
    unpack(x) = unpack_arrays(basis, x)
    packed_proj(δx, x) = pack(proj_tangent(unpack(δx), unpack(x)))

    function op(x)
        δφ = unpack(x)
        δφ = apply_sqrt_M(φ, Pks, δφ)
        pack(δφ)
    end

    Res, info = linsolve(op, pack(proj_tangent(res, φ));
                         tol=tol_krylov, verbosity=0,
                         orth=OrthogonalizeAndProject(packed_proj, pack(φ)))
    unpack(Res)
end

function apply_inv_M(basis, φ, Pks, res)
    Nk = length(Pks)

    pack(φ) = pack_arrays(basis, φ)
    unpack(x) = unpack_arrays(basis, x)
    packed_proj(δx, x) = pack(proj_tangent(unpack(δx), unpack(x)))

    function op(x)
        δφ = unpack(x)
        δφ = apply_M(φ, Pks, δφ)
        pack(δφ)
    end

    Res, info = linsolve(op, pack(proj_tangent(res, φ));
                         tol=tol_krylov, verbosity=0,
                         orth=OrthogonalizeAndProject(packed_proj, pack(φ)))
    unpack(Res)
end
