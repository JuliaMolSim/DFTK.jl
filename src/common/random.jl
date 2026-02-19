"""
Seeds the task local RNG across all MPI ranks.
A seed can be provided for reproducibility with previous runs
(for the same Julia version and Manifest.toml).
If no seed is provided, a random seed is generated on the master process.
The returned seed can be used to reproduce the run.

If any subtask is spawned, it will be seeded based on the task local RNG of its parent,
as explained in the documentation of `Random.TaskLocalRNG`.
Seeding the task local RNG at the beginning of a computation is thus sufficient.
"""
function seed_task_local_rng!(seed::Union{Nothing,Integer}, comm)
    if mpi_master(comm) && isnothing(seed)
        # Using negative seeds requires Julia 1.11 and DFTK still supports 1.10
        seed = rand(UInt64)
    end
    seed = mpi_bcast(seed, comm)
    if mpi_master(comm)
        Random.seed!(seed)
        # Generate a different seed for each process
        local_seeds = rand(typeof(seed), mpi_nprocs(comm))
        local_seed = MPI.scatter(local_seeds, comm)
    else
        local_seed = MPI.scatter(nothing, comm)
    end
    Random.seed!(local_seed)
    seed
end
