module DFTKFastGaussQuadratureExt
using FastGaussQuadrature
using DFTK
using DFTK: to_cpu, eval_kernel_fourier, VoxelAveraged
using LinearAlgebra


@views function DFTK._compute_kernel_fourier(kernel, regularization::VoxelAveraged,
                                             basis::PlaneWaveBasis{T}, qpt, q) where {T}
    model = basis.model
    q = qpt.coordinate
    
    # Get kgrid_size
    if isnothing(basis.kgrid)
        kgrid_size = Vec3{Int}(1, 1, 1)
    elseif basis.kgrid isa AbstractVector
        kgrid_size = Vec3{Int}(basis.kgrid)
    elseif basis.kgrid isa MonkhorstPack
        kgrid_size = Vec3{Int}(basis.kgrid.kgrid_size)
    else
        @error "Cannot determine kgrid_size for VoxelAveraged Coulomb model."
    end

    # Define Voxels as reciprocal cell deivided by k-mesh
    voxel_basis = model.recip_lattice * Diagonal(1 ./ Vec3{T}(kgrid_size))
    voxel_vol = abs(det(voxel_basis))
    
    # get Gauss-Legendre nodes and weights
    nodes_std, weights_std = gausslegendre(regularization.n_quadrature_points)
    # Scale from [-1, 1] to [-0.5, 0.5]
    nodes = T.(nodes_std ./ 2)
    weights = T.(weights_std ./ 2)
    
    kernel_fourier = zeros(T, length(qpt.G_vectors))

    for (iG, G) in enumerate(to_cpu(qpt.G_vectors))
        G_cart = model.recip_lattice * (G+q)

        found_singularity = (iG==1 && iszero(q))
        
        if found_singularity 
            # === At Singularity (G+q=0) use surface reduction method ===
            
            # We only do that for the 4π/(G+q)^2 kernel, hence the quadrature below
            # covers the rest if kernel_fourier is different from Coulomb().
            
            # Transforms volume integral ∫ 1/G^2 dV to surface integral Σ h * ∫ 1/G^2 dS
            integral = zero(T)
            for i in 1:3
                u_i = voxel_basis[:, i]
                u_j = voxel_basis[:, mod1(i+1, 3)]
                u_k = voxel_basis[:, mod1(i+2, 3)]
                
                # Height of the face from origin
                normal = cross(u_j, u_k)
                area_norm = norm(normal)
                
                # Calculate distance h from center to face.
                # Since face is at u_i/2, h = |(u_i/2) . n|
                h = abs(dot(u_i, normal)) / (2 * area_norm)
                
                # Integrate 1/G^2 over the face parallelogram
                face_integral = zero(T)
                for (wa, a) in zip(weights, nodes)
                    for (wb, b) in zip(weights, nodes)
                        # Parametrization of the face: r = u_i/2 + a*u_j + b*u_k
                        r_vec = 0.5f0 * u_i + a * u_j + b * u_k 
                        r_sq  = dot(r_vec, r_vec)
                        face_integral += wa * wb / r_sq
                    end
                end
                
                # Add contribution: 2 faces * h * Area * Mean(1/r^2)
                # Area factor (area_norm) comes from the Jacobian.
                integral += 2 * h * area_norm * face_integral
            end
            
            kernel_fourier[iG] = 4T(π) * (integral / voxel_vol)
        end

        # === Use smooth 3D Gaussian Quadrature ===
        integral = zero(T)
        for (wx, x) in zip(weights, nodes)
            for (wy, y) in zip(weights, nodes)
                for (wz, z) in zip(weights, nodes)
                    # q vector inside voxel
                    q_local = x * voxel_basis[:, 1] + 
                              y * voxel_basis[:, 2] + 
                              z * voxel_basis[:, 3]
                    
                    G_total = G_cart + q_local
                    Gsq = dot(G_total, G_total)

                    # switch temporarily to BigFloat
                    Gsq_big = BigFloat(Gsq)
                    val_big = eval_kernel_fourier(kernel, Gsq_big)

                    # At singularity, already captured the 4π/(G+q)^2 contribution above: subtract
                    if found_singularity
                        val_big -= 4π/Gsq_big
                    end

                    # back to type T
                    val = T(val_big)

                    integral += wx * wy * wz * val
                end
            end
        end
        kernel_fourier[iG] += integral
    end
    
    kernel_fourier
end

end
