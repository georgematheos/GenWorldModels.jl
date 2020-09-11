### A Pluto.jl notebook ###
# v0.11.12

using Markdown
using InteractiveUtils

# ╔═╡ b70f8c80-eee2-11ea-362e-77ba574cf609
using Pkg; Pkg.activate("..");

# ╔═╡ 1f947702-eee3-11ea-2b98-afa130057f1b
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

# ╔═╡ 34fbf9f8-eee3-11ea-1c73-77d84798edd4
ingredients("main.jl")

# ╔═╡ Cell order:
# ╠═b70f8c80-eee2-11ea-362e-77ba574cf609
# ╠═1f947702-eee3-11ea-2b98-afa130057f1b
# ╠═34fbf9f8-eee3-11ea-1c73-77d84798edd4
