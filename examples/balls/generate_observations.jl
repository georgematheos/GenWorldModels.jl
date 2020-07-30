module BallObservationGenerator
    using Gen
    function generate_observations(num_colors, num_samples)
        samples = [uniform_discrete(1, num_colors) for _=1:num_samples]
        println()
        println("BLOG observations:")
        for (i, sample) in enumerate(samples)
            println("obs ObsColor(draws[$(i-1)] = colors[$(sample-1)]")
        end
        println()

        println("Gen observations:")
        println("observations = choicemap(")
        for (i, sample) in enumerate(samples)
            suffix = i == length(samples) ? "" : ","
            println("\t(:samples => $i, $sample)" * suffix)
        end
        println(")")
        println("model_args = ($num_colors, $num_samples)")
    end
    generate_observations(10, 6)
end