using Plots

function make_graphs()
    dicts = Dict(
        "smart" => Dict(i=>[] for i=2:10),
        "dumb" => Dict(i=>[] for i=2:10),
        "sdds" =>Dict(i=>[] for i=2:10)
    )

    println("gonna read all files in ", joinpath(@__DIR__, "saves"))
    for filename in readdir(joinpath(@__DIR__, "saves"))
        (_, date_hour, min, sec, rest) = split(filename, "_")
        if rest in ("smart", "dumb", "sdds")
            add_data_for_file!(dicts, filename)
        end
    end

    lengths = Dict(#averages = Dict(
        "smart" => Dict(),
        "dumb" => Dict(),
        "sdds" =>Dict()
    )
    for type in ("smart", "dumb", "sdds")
        for i=2:10
            vecs = dicts[type][i]
            push!(lengths[type], i => length(vecs))
         #   average = sum(vecs)/length(vecs)
           # push!(averages[type], i => average)
        end
    end
    display(lengths)

    # for i=2:10# i=2:10
    #     smarts = averages["smart"][i] 
    #     dumbs = averages["dumb"][i] 
    #     sddss = averages["sdds"][i]
    #     xs = collect(10:10:1000)
    #     plot(xs, smarts, label="smart")
    #     plot!(xs, dumbs, label="dumb")
    #     plot!(xs, sddss, label="sdds")
    #     xlabel!("iterations")
    #     ylabel!("error in number of inferred facts")

    #     savefig(joinpath(@__DIR__, "figures", "$(i)_rels.png"))
    # end
end

function add_data_for_file!(dicts, filename)
    lines = readlines(joinpath(@__DIR__, "saves", filename))
    type = lines[1]
    num_rels = parse(Int, lines[2])
    num_facts = parse(Int, lines[3])
    inferences = map(x -> parse(Int, x), split(lines[4][5:end-1], ","))


    diffs = abs.(inferences .- num_facts)
    push!(dicts[type][num_rels], diffs)
end

make_graphs()