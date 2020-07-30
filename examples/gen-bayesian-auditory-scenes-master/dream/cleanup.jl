using Printf


println("Loading in command-line args...")
#Load in command-line arguments
dataset_name = ARGS[1]
println("Dataset name: ", dataset_name)
parallel_run_idx = ARGS[2]
println("Parallel run index: ", parallel_run_idx)

dream_path = "/om2/user/mcusi/gen-bayesian-auditory-scenes/dreams/"
parallel_name = @sprintf("%02d",parse(Int,parallel_run_idx)); 
dream_path = string(dream_path, dataset_name, "_", parallel_name, "/")
dream_files = readdir(dream_path)

startidxfn = string(dream_path, "startidx.json")
println("Continuing old run: ", startidxfn)
f=open(startidxfn,"r")
dt = read(f, String)
start_idx = parse(Int, dt)
close(f)

println("Starting clean-up!")
for sample_idx = 1:start_idx

	sample_name = @sprintf("%08d",sample_idx); 
	strchk(s) = occursin(sample_name, s)
	kfs = filter(strchk, dream_files)
	if length(kfs) > 8
		println("Too many of: ", sample_name) 
		for kf in kfs
			rm(string(dream_path, kf))
		end
	elseif 0 < length(kfs) < 8
		println("Too few of: ", sample_name)
		for kf in kfs
			rm(string(dream_path, kf))
		end
	# elseif length(kfs) == 0 #no files of this number (that's fine!)
	# 	println("Completely missing ", sample_name)
	# elseif length(kfs) == 8
	# 	# println(sample_name, " complete!")
	end

end 
println("Finished clean-up!")