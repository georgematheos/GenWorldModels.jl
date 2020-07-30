source_params=Dict(
    #KEEP IN THIS ORDER(it'll just make things less confusing): "tone", "noise", "harmonic"
    "types"=>["tone"], 
    "n_harmonics"=>30,
    "n_sources"=>Dict("type"=>"max","val"=>2),
    "n_elements"=>Dict("type"=>"max","val"=>2),
    "tp"=>Dict(
        "wait"=>Dict("a"=>Dict("dist"=>gamma,"args"=>(2.0,1.0)),
                     "mu"=>Dict("dist"=>log_normal, "args"=>(-2.0, 1.5))),
        "dur_minus_min"=>Dict("a"=>Dict("dist"=>gamma, "args"=>(2.0, 1.0)),
                    "mu"=>Dict("dist"=>log_normal, "args"=>(-2.0, 1.5)))

        ),
    "gp"=>Dict(
        "erb"=>Dict("mu"=>Dict("dist"=>uniform,"args"=>(freq_to_ERB(20.0), freq_to_ERB(9999.0))),
                    "scale"=>Dict("dist"=>gamma,"args"=>(0.5,1)),
                    "sigma"=>Dict("dist"=>gamma,"args"=>(3,1)),
                    "epsilon"=>Dict("dist"=>gamma,"args"=>(1,0.1))),
        "amp"=>Dict(
            "1D"=>Dict("mu"=>Dict("dist"=>normal,"args"=>(50, 15)),
                    "scale"=>Dict("dist"=>gamma,"args"=>(0.5,1)),
                    "sigma"=>Dict("dist"=>gamma,"args"=>(7,1)),
                    "epsilon"=>Dict("dist"=>gamma,"args"=>(1,1))),
            "2D"=>Dict("mu"=>Dict("dist"=>normal,"args"=>(10, 10)),
                    "scale_t"=>Dict("dist"=>gamma,"args"=>(0.5,2)),
                    "scale_f"=>Dict("dist"=>gamma,"args"=>(2,2)),
                    "sigma"=>Dict("dist"=>gamma,"args"=>(10,2)),
                    "epsilon"=>Dict("dist"=>gamma,"args"=>(1,1)))
            )
        ),
    "enable_switch"=>false, #inference move to switch source_types
    "duration_limit"=>2.0, #upper limit on element duration in seconds
    "dream"=>false #generating data or inference?
    )

steps = Dict("t"=>0.020, "min"=> 0.025, "f"=>4, "scale"=>"ERB"); 

gtg_params=Dict("twin"=>0.025, "thop"=>0.010, "nfilts"=>64, "fmin"=>20, "width"=>0.50, 
    "log_constant"=>1e-80, "dB_threshold"=>20, "ref"=>1e-6, "plot_max"=>100)

obs_noise = Dict("type"=>"fixed","val"=>1.0)

p = (source_params, steps, gtg_params, obs_noise)