

using LinearAlgebra
using Plots
using FastGaussQuadrature
using LinearAlgebra

include("Wave_SEM_1d.jl")
using .SEM_wave_1d


include("MMS.jl")
using .MMS


function main()

    ##########################
    #set up the time interval, the number of elements, the degree of the interpolation polynomials

    Tend = 10.0
    nsteps = 10000

    xl = 1
    xr = 7
    NumberOfNodes = 256
    nodes = zeros(NumberOfNodes)

    range = LinRange(xl, xr, NumberOfNodes)

    for i = 1:NumberOfNodes
        nodes[i] = range[i]
    end

    N = 5 #number of points we interpolate in for each element


    #set up some inital conditions and the values of the driving term f
    quadpoints, ~ = gausslobatto(N)
    totalPointCount = (length(nodes)-1) * (N-1) + 1

    x = zeros(totalPointCount)
    fVals = zeros(totalPointCount)
    uStart = zeros(totalPointCount)
    uStartDer = zeros(totalPointCount)

    omega = 6.0

    epsilon = 0.1

    for k = 1:length(nodes)-1
        for i = 1:N-1

            x[(k-1)*(N-1) + i] = nodes[k] + (1 + quadpoints[i])*(nodes[k+1]-nodes[k])/2 
    

            #f is some square wave, for instance
            #=
            if (abs((k-1)*(N-1) + i - Int(floor(length(nodes)/2))) < 20)
                fVals[(k-1)*(N-1) + i] = 10
            end
            =#

            fVals[(k-1)*(N-1) + i] = 50*exp(-1/(1-min(1, (x[(k-1)*(N-1) + i] - nodes[Int(floor(length(nodes)/3))])^2/(2*epsilon)^2)))

            #u begins as a standard mollifier
            uStart[(k-1)*(N-1) + i] = 5*exp(-1/(1-min(1, (x[(k-1)*(N-1) + i] - nodes[Int(floor(length(nodes)/2))])^2/epsilon^2)))

        end
    end

    x[end] = nodes[end]


    animate = true
    snapshotFrequency = 100

    #SEM_wave_1d.simulate(nodes, nsteps, Tend, N, fVals, omega, uStart, animate, snapshotFrequency)

    simul = SEM_wave_1d.SEM_Wave(nodes, nsteps, Tend, N, fVals, omega)

    simul.useMMS = true
    fVals = SEM_wave_1d.ForcingTerm2(simul, 25)


    N = simul.PointsPerElement     #number of quadrature points
    K = length(simul.nodes)-1      #number of elements
    degreesOfFreedom = K*(N-1) + 1

    u_xx = zeros(degreesOfFreedom)


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
        
            SEM_wave_1d.setDegreesOfFreedom!(simul, k, simul.uNow, vals_k, false)
            SEM_wave_1d.setDegreesOfFreedom!(simul, k, u_xx, vals_k2, false)

        end


    end

    lapVals = SEM_wave_1d.LaplaceCalculation3(simul, simul.uNow)

    plt = plot(x[2:end-1], u_xx[2:end-1]-lapVals[2:end-1])

    #println(fVals)

    #SEM_wave_1d.simulate2(nodes, nsteps, Tend, N, fVals, omega, uStart, uStartDer, animate, snapshotFrequency)

end