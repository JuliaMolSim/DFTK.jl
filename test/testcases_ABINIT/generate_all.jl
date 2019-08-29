function run_folder(folder)
    cd(folder) do
        if !isfile("extra.jld") || !isfile("out_GSR.nc")
            println("#\n#--  Running $folder\n#")
            include(joinpath(folder, "generate.jl"))
        end
    end
end

for file in readdir()
    if isdir(file) && isfile(joinpath(file, "generate.jl"))
        run_folder(file)
    end
end
