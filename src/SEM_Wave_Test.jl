

using LinearAlgebra
using Plots
using FastGaussQuadrature
using LinearAlgebra

include("SEM_Wave_1d.jl")
using .SEM_Wave_1d


include("MMS.jl")
using .MMS



function main()

    ##########################
    #set up the time interval, the number of elements, the degree of the interpolation polynomials

    Tend = 6.0
    nsteps = 9000

    xl = -1
    xr = 1
    NumberOfNodes = 321
    nodes = zeros(NumberOfNodes)
    
    range = LinRange(xl, xr, NumberOfNodes)
    
    for i = 1:NumberOfNodes
        nodes[i] = range[i]
    end

    N = 5 #number of points we interpolate in for each element
    

    #set up some inital conditions and the values of the driving term f
    quadpoints, ~ = gausslobatto(N)
    totalPointCount = (length(nodes)-1) * (N-1) + 1

    x = SEM_Wave_1d.ConstructX(nodes, quadpoints)
    fVals = zeros(totalPointCount)
    uStart = zeros(totalPointCount)
    uStartDer = zeros(totalPointCount)

    omega = 6.0

    epsilon = 0.1

    for k = 1:length(nodes)-1
        for i = 1:N-1

            fVals[(k-1)*(N-1) + i] = 50*exp(-1/(1-min(1, (x[(k-1)*(N-1) + i] - nodes[Int(floor(length(nodes)/3))])^2/(2*epsilon)^2)))

            #u begins as a standard mollifier
            uStart[(k-1)*(N-1) + i] = 5*exp(-1/(1-min(1, (x[(k-1)*(N-1) + i] - nodes[Int(floor(length(nodes)/2))])^2/epsilon^2)))

        end
    end

    animate = true
    snapshotFrequency = 100

    #SEM_Wave_1d.simulate(nodes, nsteps, Tend, N, fVals, omega, uStart, animate, snapshotFrequency)
    
    simul = SEM_Wave_1d.SEM_Wave(nodes, nsteps, Tend, N, fVals, omega)

    simul.useMMS = true
    fVals = SEM_Wave_1d.ForcingTerm(simul, 25)


    N = simul.PointsPerElement     #number of quadrature points
    K = length(simul.nodes)-1      #number of elements
    degreesOfFreedom = K*(N-1) + 1
    
    u_xx = zeros(degreesOfFreedom)


    
    #this bit of code shows calculates and plots the difference between the Laplacian found symbolically (using MMS.jl) and the one found using SEM_Wave
    
    if simul.useMMS

        x_loc = zeros(N)

        z = (1 .+ simul.QuadPoints)

        vals_k = zeros(N)
        vals_k2 = zeros(N)
        t = 0.0

        for k = 1:K

            delta_x_k = simul.nodes[k+1] - simul.nodes[k]
            
            x_loc .= simul.nodes[k] .+ 0.5*z*delta_x_k

            for j = 1:N
                vals_k[j] = MMS.MMSfun(x_loc[j], t, 0, 0, simul.MMS_j)
                vals_k2[j] = MMS.MMSfun(x_loc[j], t, 2, 0, simul.MMS_j)
            end
            
            SEM_Wave_1d.SetDegreesOfFreedom!(simul, k, simul.uNow, vals_k, false)
            SEM_Wave_1d.SetDegreesOfFreedom!(simul, k, u_xx, vals_k2, false)

        end


    end
    
    lapVals = SEM_Wave_1d.LaplaceCalculation(simul, simul.uNow)

    plt = plot(x[2:end-1], u_xx[2:end-1]-lapVals[2:end-1])
    


    #very confused about why res is printed when one runs this if one makes Simulate return simul.uNow 
    #out = SEM_Wave_1d.Simulate(nodes, nsteps, Tend, N, fVals, omega, uStart, uStartDer, animate, snapshotFrequency)



end


