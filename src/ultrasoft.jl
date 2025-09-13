function atom_local_grid(lattice::AbstractMatrix{T}, fft_grid::FFTGrid, position, radius) where {T}
    # Find bounds on the points we have to check along each dimension
    # Bounds from https://math.stackexchange.com/a/1230292
    AtAinv = inv(lattice'lattice)
    Atinv = inv(lattice')
    xbound = radius * AtAinv[1, 1] / norm(Atinv * [1, 0, 0])
    ybound = radius * AtAinv[2, 2] / norm(Atinv * [0, 1, 0])
    zbound = radius * AtAinv[3, 3] / norm(Atinv * [0, 0, 1])

    # +1 for inexact computation of the bound, +1 for the rounding of the atom position 
    xmax = ceil(Int, xbound * fft_grid.fft_size[1]) + 2
    ymax = ceil(Int, ybound * fft_grid.fft_size[2]) + 2
    zmax = ceil(Int, zbound * fft_grid.fft_size[3]) + 2

    grid_indices = Vec3{Int}[]
    atom_distances = Vec3{T}[]

    x0, y0, z0 = round.(position .* fft_grid.fft_size)
    for iz=z0-zmax:z0+zmax, iy=y0-ymax:y0+ymax, ix=x0-xmax:x0+xmax
        atom_distance = lattice * ([ix, iy, iz] ./ fft_grid.fft_size - position)

        if norm(atom_distance) <= radius
            # mod in case we wrap around grid boundaries
            index = mod.([ix, iy, iz], fft_grid.fft_size) .+ 1
            push!(grid_indices, index)
            push!(atom_distances, atom_distance)
        end
    end

    (; grid_indices, atom_distances)
end

function precompute_augmentation_regions(model::Model{T}, fft_grid::FFTGrid) where {T}
    # TODO: hardcoded for the Si psp I have
    psp = model.atoms[1].psp
    # proj_indices = [1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6]
    # proj_to_m = [0, 0, -1, 0, 1, -1, 0, 1, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2]
    proj_indices = [1, 2, 3, 4, 3, 4, 3, 4, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6]
    proj_to_m = [0, 0, -1, -1, 0, 0, 1, 1, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2]

    gaunt_coefs = gaunt_coefficients(T, psp.lmax)

    map(enumerate(model.positions)) do (iatom, pos)
        @info "Computing grid for atom $iatom"
        # TODO: hardcoded ultrasoft_cutoff_radius (is this what this means?)
        (; grid_indices, atom_distances) = atom_local_grid(model.lattice, fft_grid, pos, 1.8)
        augmentations = zeros(T, length(grid_indices), 18, 18)
        @info "Looping augmentation region for atom $iatom: box size $(length(grid_indices))"
        for (r, Q) in zip(atom_distances, eachslice(augmentations; dims=1))
            for iproj=1:18, jproj=iproj:18
                Q[iproj, jproj] = eval_augmentation(psp, gaunt_coefs, proj_indices[iproj], proj_to_m[iproj], proj_indices[jproj], proj_to_m[jproj], r)
                if iproj != jproj # By symmetry
                    Q[jproj, iproj] = Q[iproj, jproj]
                end
            end
        end
        (; grid_indices, atom_distances, augmentations)
    end
end