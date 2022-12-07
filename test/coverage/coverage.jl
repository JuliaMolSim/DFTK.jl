using Coverage

cd(joinpath(@__DIR__, "..", "..")) do
    processed = process_folder()
    # TODO Does not yet work
    # Codecov.submit_local(processed)
    covered_lines, total_lines = get_summary(processed)
    percentage = covered_lines / total_lines * 100
    println("($(percentage)%) covered")
end
