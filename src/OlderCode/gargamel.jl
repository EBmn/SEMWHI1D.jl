
include("MMS.jl")
using .MMS


using Plots


Nx = 101
Nt = 201
x = LinRange(0, 2*pi, Nx)
t = LinRange(0, 2*pi, Nt)


MMS_j = MMS.MMS_jet(1, 2)


u = zeros(Nx, Nt)

for j = 1:Nt    
    for i = 1:Nx
        u[i, j] = MMS.MMSfun(x[i], t[j], 2, 2, MMS_j)
    end
end


contour(t, x, u)