using StaticArrays
using LinearAlgebra
using Optim

struct ExchangeLandscape
    _0111::ComplexF64
    _0001::ComplexF64
    _0110::ComplexF64
    _0101::ComplexF64
    _0011::ComplexF64
    _0000::ComplexF64
    _1111::ComplexF64
    _i00i::ComplexF64
    _i11i::ComplexF64
    _i01i::ComplexF64
end

function ExchangeLandscape(basis, ψ, ik, _0, _1)
    
    _0111 = integral(basis, ψ, ik, _0, _1, _1, _1)
    _0001 = integral(basis, ψ, ik, _0, _0, _0, _1)
    _0110 = integral(basis, ψ, ik, _0, _1, _1, _0)
    _0101 = integral(basis, ψ, ik, _0, _1, _0, _1)
    _0011 = integral(basis, ψ, ik, _0, _0, _1, _1)
    _0000 = integral(basis, ψ, ik, _0, _0, _0, _0)
    _1111 = integral(basis, ψ, ik, _1, _1, _1, _1)

    _i00i = sum([integral(basis, ψ, ik, _i, _0, _0, _i) for _i in 1:(_0-1)])
    _i11i = sum([integral(basis, ψ, ik, _i, _1, _1, _i) for _i in 1:(_0-1)])
    _i01i = sum([integral(basis, ψ, ik, _i, _0, _1, _i) for _i in 1:(_0-1)])

    println("<01|11>: ", _0111)
    println("<00|01>: ", _0001)
    println("<01|10>: ", _0110)
    println("<01|01>: ", _0101)
    println("<00|11>: ", _0011)
    println("<00|00>: ", _0000)
    println("<11|11>: ", _1111)
    println("<i0|0i>: ", _i00i)
    println("<i1|1i>: ", _i11i)
    println("<i0|1i>: ", _i01i)
    println()

    ExchangeLandscape(_0111, _0001, _0110, _0101, _0011, _0000, _1111, _i00i, _i11i, _i01i)
end

function (L::ExchangeLandscape)(θ, φ)
    rot_K_00 = 0.0
    rot_K_00 += 4 * sin(θ) * cos(θ)^(3) * cos(φ) * real(L._0111) 
    rot_K_00 += 4 * sin(θ) * cos(θ)^(3) * sin(φ) * imag(L._0111)
    rot_K_00 += 4 * sin(θ)^(3) * cos(θ) * cos(φ) * real(L._0001) 
    rot_K_00 += 4 * sin(θ)^(3) * cos(θ) * sin(φ) * imag(L._0001) 
    rot_K_00 += 2 * cos(θ)^(2) * sin(θ)^(2) * (L._0110 + L._0101)
    rot_K_00 += 2 * cos(θ)^(2) * sin(θ)^(2) * cos(2*φ) * real(L._0011)
    rot_K_00 += 2 * cos(θ)^(2) * sin(θ)^(2) * sin(2*φ) * imag(L._0011)
    rot_K_00 += cos(θ)^(4) * L._1111
    rot_K_00 += sin(θ)^(4) * L._0000

    sum_rot_K_i0 = 0.0
    sum_rot_K_i0 += 2 * cos(θ) * sin(θ) * cos(φ) * real(L._i01i) 
    sum_rot_K_i0 += 2 * cos(θ) * sin(θ) * sin(φ) * imag(L._i01i)
    sum_rot_K_i0 += cos(θ)^(2) * L._i11i 
    sum_rot_K_i0 += sin(θ)^(2) * L._i00i

    sum_K_i0 = 1.0 * L._i00i
    K_00     = 1.0 * L._0000 
    
    Δ = - 2 * (sum_rot_K_i0 - sum_K_i0) - (rot_K_00 - K_00) 
    @assert abs(Δ) ≈ abs(real(Δ))
    
    real(Δ)
end

function h_kl(basis_core, scfres, ik, low, high)
    n = high - low + 1
    ψ = deepcopy(scfres.ψ)
    occupation = deepcopy(scfres.occupation)
    occupation[1][1:low-1] .= 0.0
    _, ham = energy_hamiltonian(basis_core, ψ, occupation)
    Hψ = (ham.blocks[1] * ψ[1])
    h_kl = ψ[1]' * Hψ
    return h_kl[low:high,low:high]
end


function off_block_integrals_expensive(basis, ψ, ik, low, high)
    n_active = high-low+1
    T = in_block_integrals(basis, ψ, ik, 1, high)
    integrals = zeros(ComplexF64, n_active, n_active)
    for k in 1:low-1
        for i in 1:n_active
            for j in 1:n_active
                integrals[i,j] += -4 * T[k,i+low-1,k,j+low-1] + 2 * T[k,i+low-1,j+low-1,k]
            end
        end
    end
    return integrals
end


function off_block_integrals(basis, ψ, ik, low, high)
    n_active = high-low+1
    n_passive = low-1

    ψrs = [ifft(basis, basis.kpoints[ik], ψ[ik][:,i]) for i in 1:high]
    r_grid = length(ψrs[1])

    Gsq = [sum(abs2, basis.model.recip_lattice * G) for G in G_vectors(basis)]
    kernel = [g ≈ 0 ? 0.0 : 4π / g for g in Gsq]

    D = Matrix{ComplexF64}(undef, r_grid, n_active)
    V = Matrix{ComplexF64}(undef, r_grid, n_active)
    integrals = zeros(ComplexF64, n_active, n_active)

    D1 = Matrix{ComplexF64}(undef, r_grid, 1)
    V1 = Matrix{ComplexF64}(undef, r_grid, 1)

    for k in 1:low-1
        idx = 1
        for i in low:high
            ρ = conj.(ψrs[i]) .* ψrs[k]
            D[:, idx] = vec(ρ)
            ρg = fft(basis, ρ)
            V[:, idx] = vec(ifft(basis, kernel .* ρg))
            idx += 1
        end
        integrals += - (adjoint(D) * V) .* basis.dvol

        ρk = conj.(ψrs[k]) .* ψrs[k] 
        ρkg = fft(basis, ρk)
        V1[:,1] = vec(ifft(basis, kernel .* ρkg))

        for i in low:high
            for j in low:high
                 
                D1[:,1]  = vec(conj.(ψrs[i]) .* ψrs[j])
                integrals[j-low+1,i-low+1] += (2 * adjoint(D1) *  V)[1,1] .* basis.dvol
            end
        end

    end
    

    #D1 = Matrix{ComplexF64}(undef, n_active * n_active, r_grid)
    #V1 = Matrix{ComplexF64}(undef, r_grid, 1)

    #ρk = zeros(ComplexF64, size(ψrs[1])...)
    #for k in 1:low-1
    #    ρk += conj.(ψrs[k]) .* ψrs[k]
    #end

    #idx = 1
    #for i in 1:n_active, j in 1:n_active
    #    ρ = conj.(ψrs[i]) .* ψrs[j]
    #    D1[idx, :] = vec(ρ)
    #    idx += 1
    #end

    #ρkg = fft(basis, ρk)
    #V1[:, 1] = vec(ifft(basis, kernel .* ρkg))

    #integrals_flat = (D1 * V1) .* basis.dvol
    #integrals += 2 * reshape(integrals_flat, n_active, n_active)

    return integrals
end

function in_block_integrals(basis, ψ, ik, low, high)
    n_active = high-low+1

    ψrs = [ifft(basis, basis.kpoints[ik], ψ[ik][:,i]) for i in low:high]
    r_grid = length(ψrs[1])

    Gsq = [sum(abs2, basis.model.recip_lattice * G) for G in G_vectors(basis)]
    kernel = [g ≈ 0 ? 0.0 : 4π / g for g in Gsq]

    D = Matrix{ComplexF64}(undef, r_grid, n_active*n_active)
    V = Matrix{ComplexF64}(undef, r_grid, n_active*n_active)

    idx = 1
    for k in 1:n_active, i in 1:n_active
        ρ = conj.(ψrs[i]) .* ψrs[k]
        D[:, idx] = vec(ρ)
        ρg = fft(basis, ρ)
        V[:, idx] = vec(ifft(basis, kernel .* ρg))
        idx += 1
    end

    integrals_flat = (transpose(D) * V) .* basis.dvol
    integrals = reshape(integrals_flat, n_active, n_active, n_active, n_active)
    integrals = permutedims(integrals, (1,3,2,4))
    return integrals
end

function integral(basis, ψ, ik, i, j, k, l)
    ψi = ψ[ik][:,i]
    ψj = ψ[ik][:,j]
    ψk = ψ[ik][:,k]
    ψl = ψ[ik][:,l]

    ψri = ifft(basis, basis.kpoints[ik], ψi)
    ψrj = ifft(basis, basis.kpoints[ik], ψj)
    ψrk = ifft(basis, basis.kpoints[ik], ψk)
    ψrl = ifft(basis, basis.kpoints[ik], ψl)

    ρr_ik = conj.(ψri) .* ψrk
    ρr_jl = conj.(ψrj) .* ψrl

    Gsq = [sum(abs2, basis.model.recip_lattice * G) for G in G_vectors(basis)]
    kernel = [g ≈ 0 ? 0.0 : 4π / g for g in Gsq]

    ρg_jl = fft(basis, ρr_jl) 

    V_g = kernel .* ρg_jl
    V_r = ifft(basis, V_g)
    ijkl = sum(ρr_ik .* V_r) * basis.dvol
    return ijkl
end

function check_in_block_integrals(basis, ψ, ik, low, high)
    n = high - low + 1
    fast = in_block_integrals(basis, ψ, ik, low, high)

    slow = zeros(ComplexF64, n, n, n, n)
    for i in low:high
    for j in low:high
    for k in low:high
    for l in low:high
        iind = i-low+1
        jind = j-low+1
        kind = k-low+1
        lind = l-low+1
        slow[iind,jind,kind,lind] = integral(basis, ψ, ik, i,j,k,l)
    end
    end
    end
    end
    
    println(isapprox(slow, fast))
end

function check_off_block_integrals(basis, ψ, ik, low, high)
    n = high - low + 1
    fast = off_block_integrals_expensive(basis, ψ, ik, low, high)

    slow = zeros(ComplexF64, n, n)
    for i in low:high
    for j in low:high
        iind = i-low+1
        jind = j-low+1
    for k in 1:low-1
        slow[iind,jind] +=  2 * integral(basis, ψ, ik, k,i,j,k) - 4 * integral(basis, ψ, ik, k,i,k,j)
    end
    end
    end
    
    println(slow[:,:])
    println(fast[:,:])

    println(isapprox(slow, fast))
end

function local_solver(L::ExchangeLandscape, θ0, φ0)
    result = optimize(p -> L(p[1],p[2]), [θ0,φ0], BFGS())
    θopt, φopt = Optim.minimizer(result)

    return θopt, φopt
end

function global_grid_solver(L::ExchangeLandscape, Nθ, Nφ)
    θs = range(0.0, π/2,   length=Nθ)
    φs = range(0.0, π,  length=Nφ)

    emap = [L(θ,φ) for θ in θs, φ in φs]
    startguess = argmin(emap)

    θ0 = θs[startguess[1]]
    φ0 = φs[startguess[2]]

    return θ0, φ0
end

function global_local_solver(L::ExchangeLandscape, Nθ, Nφ)
    return local_solver(L,global_grid_solver(L, Nθ, Nφ)...)
end

function R_SU2(θ,φ)
    # ── Erste Spalte: Punkt auf ℂP¹ ──
    e₁ = @SVector [
        sin(θ) * cis(φ),
        cos(θ) + 0im
       ]

    e₂ = @SVector [
        - cos(θ) + 0im,
        sin(θ) * cis(-φ)
       ]

    return @SMatrix [e₁[1] e₂[1];
                     e₁[2] e₂[2]]
end
