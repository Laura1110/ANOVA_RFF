using Distributions
using Combinatorics
using IterativeSolvers
using SpecialFunctions
#using PlotlyJS
using LinearAlgebra
using Plots 
using CSV 
using DataFrames
using Copulas
using ANOVAapprox
#pyplot()

#using Plots
#plotlyjs()

include("../anova_RFF.jl")
function fun(X,f)   
    if f ==1
        #return X[4,:].^2 ./ (1 .+X[4,:].^2 ).^2 .+  X[3,:] .* X[2,:] ./ ( (1 .+X[2,:].^2 ) .* (1 .+X[3,:].^2 )) .+  X[1,:] .* X[2,:] ./ ( (1 .+X[1,:].^2 ) .* (1 .+X[2,:].^2 )).+  exp.(-X[4,:])      #fT1
        return X[4,:].^2 .+  X[3,:] .* X[2,:] .+  X[1,:] .* X[2,:] .+  X[4,:]      #fT1
        #return max.(1/9 .- X[1,:].^2) .* max.(1/9 .- X[2,:].^2) + max.(1/9 .- X[4,:].^2) + max.(1/9 .- X[2,:].^2) .* max.(1/9 .- X[3,:].^2)
    elseif f ==2
        	return sin.(X[1,:]) + 7 .* sin.(X[2,:]).^2 + 0.1.* X[3,:].^4 .*  sin.(X[1,:]) #fT2
    elseif f ==3
        return 10 .* sin.(pi .*  X[1,:] .* X[2,:]) + 20 .* (X[3,:]  .- 0.5).^2 + 10 .* X[4,:]  + 5 .*  X[5,:]   #fT3
    elseif f == 4
        return X[4,:] .+  X[3,:] .+  X[2,:] .+  X[1,:]    
    elseif f == 5
        return 2\ sqrt(6) .* (- max.(X[1,:],X[2,:]) - max.(X[3,:],X[4,:]) - max.(X[5,:],X[6,:]) .+1)      #fT4
    end
 
end


#### RFF  ######


f = 3

algo = "anova"
#algo = "rff"

println("f= ",f)
 
println("algo = ",algo)

if f ==1
    sample_dist =Uniform(0,1)
    d = 20 
    q = 2
    epsilon = 0.003
    U_echt = [Int.([1,2]),Int.([2,3]),Int.([4])]
    anova_step= "ascent"
elseif f == 2
    sample_dist = Uniform(-pi,pi)
    d = 5
    q = 2
    epsilon = 0.005                                                     # bei 0.01 findet es die 3 nicht
    U_echt = [Int.([1]),Int.([2]), Int.([1,3]), Int.([3])]
    anova_step = "ascent"
elseif f ==3
    sample_dist = Uniform(0,1)
    d = 10 
    q = 2
    epsilon = [0.001,0.05]
    U_echt = [Int.([1,2]),Int.([3]),Int.([4]), Int.([5])]
    anova_step = "ascent"
elseif f ==5
    sample_dist = Uniform(-0.5,0.5)
    d = 10 
    q = 2
    epsilon = [0.01,0.05]
    U_echt = [Int.([1,2]),Int.([3,4]),Int.([5,6])]
    anova_step = "descent"
end



dist_rrf = 1
if dist_rrf == 1
    dd = Normal(0,1/q)
else
    dd = Cauchy(0,1/q)
end


if algo == "rff"
    Ms = [500, 1000]
    #Ms = [200]
    global mse_s = Dict()   
    global mse_as = Dict() 
    global mse_U = Dict() 
    global mse_h = Dict() 
    global mse_ah = Dict() 
        
    anz_mse = 1

    for M in Ms
        println("--------")
        println("M:",M)
        N = Int(ceil(5*M))
        M_test = M

        global mse_s[M] = 0  
        global mse_as[M] = 0  
        global mse_U[M] = 0  
        global mse_h[M] = 0   
        global mse_ah[M] = 0 
        for kk = 1:anz_mse
            X = anova_RFF.make_X(d,M,dist = sample_dist)
            X_test = anova_RFF.make_X(d,M,dist = sample_dist)
            
            y = Complex.(fun(X,f))
            y_test = Complex.(fun(X_test,f))
            if anova_step == "descent"
                U = collect(combinations(1:d,q))
            else
                U = collect(combinations(1:d,1))
            end
            #U = vcat(U, [[1]], [[2]])
            #U = [[1],[2],[4],[3], [5,6], [1,2]]
            
            #dist = dd
            
            
            solver = "l2"
            l = 1e-12
            println(1)
            shr = anova_RFF.RFF_model(X,y, "exp")
            UU = anova_RFF.ANOVA_boosting(shr,q,N, dependence = false, anova_step = anova_step, epsilon = epsilon, l_w = 0.1, dist = dd, verbose = true)
            println("found the ANOVA index set U: ", UU)
            UU = anova_RFF.anti_downward_closed(UU)
            println("anti dwc U: ", UU)
            shr = anova_RFF.shrimp(X,y, UU, N, l= l,dist = dd, verbose = false)
            global mse_as[M] += anova_RFF.get_mse(shr, X_test,y_test) /anz_mse            
            println("mse end with anova: ",mse_as[M])
           


            println(2)
            U = collect(combinations(1:d,q))
            shr2 = anova_RFF.RFF_model(X,y, "exp")
            shr2 = anova_RFF.shrimp(X,y, U, N, l= l,dist = dd, verbose = false)
            global mse_s[M] += anova_RFF.get_mse(shr2, X_test,y_test) / anz_mse
            println("mse end without anova: ",mse_s[M])
            
            

            # println(3)
            # U = U_echt
            # shr3 = anova_RFF.RFF_model(X,y,U,N,"exp",  dd)
            # shr3 = anova_RFF.shrimp(X,y, U, N, l= l,dist = dd)
            # global mse_U[M] += anova_RFF.get_mse(shr3, X_test,y_test) / anz_mse
            # println("mse end correct U: ",mse_U[M])


            
            
        end
        
        
        CSV.write("numerik_indep_rff2.csv",DataFrame(d = d, q = q, M = M ,dist_rrf = dist_rrf, f=f, s = mse_s[M] ,as =  mse_as[M]),  append=true)
    end
            

elseif algo == "anova"  #### ANOVA approx####
    
    using GroupedTransforms
    ns = collect(1:1:4)
    #λs = [0.0,1.0,10.0,100.0,0.001,0.0001]
    λs = [0.0]

    mse = Dict()
    mse_a = Dict()
    
    for M in [500,1000]
        println("--------")
        #n=1
        println("M:",M)

        #M = Int(ceil(N .*log(N)))
        #M = N 
        
        M_test = M
        X = anova_RFF.make_X(d,M,dist = sample_dist)
        X_test = anova_RFF.make_X(d,M,dist = sample_dist)

        y = fun(X,f)
        y_test = fun(X_test,f)

        if M == 500
            if f == 1
                n = 3
            else
                n = 2
            end
        else 
            n = 3
        end

        ### transform the samples to the torus:
        #if f == 1
            # eta = 1/(2^(n+1))
            # global X = 1/2 .* erf.(X./sqrt(2)) 
            # global X_test = 1/2 .* erf.(X_test./sqrt(2)) 
            # global X =  (1-eta) .* (X.+0.5) .- 0.5 
            # global X_test =  (1-eta) .* (X_test.+0.5) .- 0.5 
        if f==2
            global X = X ./ pi .* 0.5
            global X_test = X_test ./ pi .* 0.5
        elseif f==3 || f==1 
            eta = 1/(2^(n+1))

            global X = X .- 0.5
            global X_test = X_test .- 0.5
            global X =  (1-eta) .* (X.+0.5) .- 0.5 
            global X_test =  (1-eta) .* (X_test.+0.5) .- 0.5 

        end



        
        ### parameter for ANOVAapprox::

 
            if f==1 
                anova_step = "ascent"
                epsilon = [0.03,0.01]
            elseif f == 2
                anova_step ="descent"
                epsilon = [0.01,0.01]
            elseif f == 3
                anova_step ="ascent"
                epsilon = [0.03,0.01]
            elseif f == 5
                anova_step ="ascent"
                epsilon = [0.1,0.04]
            end

        if anova_step == "descent"

            ff = ANOVAapprox.approx(X, y, q, Int.(n.*ones(q)), "chui2")
            ANOVAapprox.approximate(ff, lambda = λs)

            mse[M] = minimum(values(ANOVAapprox.get_mse(ff, X_test,y_test)))
            #y_eval = ANOVAapprox.evaluate(ff,X_test)
            global gsis = ANOVAapprox.get_GSI(ff,0.0)
            println("gsi:", gsis)

            U = ANOVAapprox.get_ActiveSet(ff, epsilon[1] .* ones(q), 0.0)
            println("found U:", U)
        else
            qq = 1
            ff = ANOVAapprox.approx(X, y, qq, Int.(n.*ones(qq)), "chui2")
            ANOVAapprox.approximate(ff, lambda = λs)
            global gsis = ANOVAapprox.get_GSI(ff,0.0)
            println("gsi:", gsis)
            mse[M] = minimum(values(ANOVAapprox.get_mse(ff, X_test,y_test)))
            #y_eval = ANOVAapprox.evaluate(ff,X_test)
            U = ANOVAapprox.get_ActiveSet(ff, epsilon[qq] .* ones(1), 0.0)
            println("found U:", U)
            U_new = U
            if qq < q
                U_new2 = copy(U)
                for u in U
                    for v in U
                        if in(unique(sort(vcat(u,v))),U_new2) || length(unique(sort(vcat(u,v)))) >qq+1
                        else 
                            append!(U_new2,[sort(vcat(u,v))])
                        end
                    end
                end
                U_new = U_new2
                qq = qq + 1
            end


            ff = ANOVAapprox.approx(X, y, U_new, Int.(n.*ones(qq)), "chui2")
            ANOVAapprox.approximate(ff, lambda = λs)
            mse[M] = minimum(values(ANOVAapprox.get_mse(ff, X_test,y_test)))
            #y_eval = ANOVAapprox.evaluate(ff,X_test)
            U = ANOVAapprox.get_ActiveSet(ff, epsilon[qq] .* ones(qq), 0.0)
            println("found U:", U)
            global gsis = ANOVAapprox.get_GSI(ff,0.0)
            println("gsi:", gsis)


        end
        
        
        q_new = maximum(length.(values(U)))
        global N = 1
        for u in U
            global N += CWWTtools.datalength(Int.(n.*ones(length(u))))
        end

        while N * log(N) < M && n<8
            n+=1 
            global N = 1
            for u in U
                global N += CWWTtools.datalength(Int.(n.*ones(length(u))))
            end
        end
        if f == 3 
            n+=1
        end
        println("neues n:", n)

                          
        

        
        global ff2 =  ANOVAapprox.approx(X, y, U, Int.((n).*ones(q_new)), "chui2")
        
        ANOVAapprox.approximate(ff2, lambda = λs)
        mse_a[M] = minimum(values(ANOVAapprox.get_mse(ff2, X_test,y_test)))
        println("mse:", mse_a[M])

        CSV.write("numerik_indep_rff2.csv",DataFrame(d = d, q = q, M = M ,dist_rrf = 3 , f=f, s = mse[M] ,as =  mse_a[M]),  append=true)
    end

end
#plot(X[4,:],ff.trafo[[4]] * ff.fc[0.0][[4]],line=:scatter)
