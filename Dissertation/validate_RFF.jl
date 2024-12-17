using StatsBase
using DataFrames
using ANOVAapprox
using LinearAlgebra
#using JLD2
using FileIO
using Random
using Statistics
using LibTest
using Plots
using DelimitedFiles
using MAT
using Combinatorics
using Distributions
using CSV

include("../anova_RFF.jl")

num = 1
#algo = "ASHRIMP"
algo = "AHARFE"
println("Algorithmus:  ", algo)
q = 2
#for #dataset in ["housing", "galaxy", "skillcraft", "forestfire", "airfoil", "propulsion"]
    #dataset =  "airfoil"
    dataset = "protein"
    #dataset = "forestfire"
    #dataset = "galaxy"
    #dataset = "propulsion"
    # dataset = "housing"
    #dataset = "elevators"
    #dataset = "protein"
    #for q in [1,2,3,4,5,6,7]
    

# dataset = "propulsion"
# dataset = "galaxy"
# dataset = "skillcraft"
# dataset = "airfoil"
# dataset = "forestfire"
# dataset = "housing"


        println("dataset:  ", dataset)
        println("======")

        
        # Load data:
        if dataset == "propulsion"
            vars = matread("propulsion.mat")
        elseif dataset == "galaxy"
            vars = matread("galaxy.mat")
        elseif dataset == "skillcraft"
            vars = matread("skillcraft.mat")
        elseif dataset == "airfoil"
            vars = matread("airfoil.mat")
        elseif dataset == "forestfire"
            vars = matread("forestfire.mat")
        elseif dataset == "housing"
            vars = matread("housing.mat")
        elseif dataset == "protein"
            vars = matread("protein.mat")
        elseif dataset == "elevators"
            vars = matread("elevators.mat")
        end
        X_train = Matrix(vars["Xtr"]')
        X_test = Matrix(vars["Xte"]')
        Y_train = vec(vars["Ytr"])
        Y_test = vec(vars["Yte"])

        # if dataset == "galaxy"
        #     X_train = X_train[vcat(1:5,7:12,13:19),:]
        #     X_test = X_test[vcat(1:5,7:12,13:19),:]

        # end

        
        # if dataset == "airfoil"
        #     X_train = X_train[1:5,:]
        #     X_test = X_test[1:5,:]
        # end

        d, M = size(X_train)

        # Parameters:
        if dataset == "propulsion"
            N = Int(floor(5*M   ))
            if algo == "HARFE"
                N = 3000
            end
            l = 1e-12
            epsilon = 0.01
            l_w = 1
            l_harfe = 10^(-10)
            s = 300
            anova_step = "ascent"
            dist = Normal(0,1)

        elseif dataset == "galaxy"
            N = Int(floor(5*M   ))
            if algo == "HARFE"
                N = 10000
                s = 1000
            end
            if algo == "AHARFE"
                s = 500
            end
            l = 1e-6
            epsilon = 0.00001
            l_harfe = 10^(-7)
            #s = 1000
            l_w = 100
            anova_step = "ascent"
            dist = Normal(0,1)


        elseif dataset == "airfoil"
            N = Int(floor(5*M   ))
            if algo == "HARFE" 
                N = 80000
            end
            l = 1e-6
            epsilon = 0.01
            if algo == "AHARFE"
                epsilon = 0.05
            end
            l_w = 100
            l_harfe = 1
            s = 5000
            anova_step = "ascent"
            dist = Normal(0,1/q)

        elseif dataset == "forestfire"
            N = Int(floor(10*M   ))   # bei shrimp eher 5*M, bei Harfe 20*M
            if algo == "HARFE"
                N = 4220
            end
            l = 1e-6
            epsilon = 0.01
            l_w = 100
            l_harfe = 0.5
            s = 422
            anova_step = "ascent"
            dist = Normal(0,1/q)
            #dist = Uniform(-1,1)
            #dist = Cauchy(0,1)

        elseif dataset == "housing"
            N = Int(floor(10*M   ))
            l = 1e-6
            l_w = 100
            epsilon = 0.01
            l_harfe = 1
            s = 100
            anova_step = "descent"
            dist = Normal(0,1/q)
            if algo == "HARFE"
                N = 10000
            end
            if algo == "SHRIMP"
                q = 7
            end
            if algo == "ASHRIMP"
                q = 1
                epsilon =1
                l_w = 100
                dist = Cauchy(0,1.5)
            end
            

        elseif dataset == "protein"
            N = Int(floor(M/5))
            if algo == "HARFE"
                N = 4000
            end
            l = 1e-12
            epsilon = 0.01
            l_w = 10
            l_harfe = 1
            s = 1000
            anova_step = "ascent"
            dist = Normal(0,1/q)
            q = 3

        elseif dataset == "elevators"
            N = Int(floor(M/5 ))
            l = 1e-12
            l_w = 10
            epsilon = 0.05
            l_harfe =0.0001
            anova_step = "ascent"
            dist = Normal(0,1/q)
            s= 1000
            
        end
        #

    

        
        if algo == "SHRIMP" || algo == "HARFE"
            U = collect(combinations(1:d,q))
        elseif anova_step == "ascent"
            U = reduce(vcat,[collect(combinations(1:d,qq)) for qq = 0:1])
        elseif algo == "ASHRIMP"|| algo == "AHARFE"
            U = reduce(vcat,[collect(combinations(1:d,qq)) for qq = 0:q])
        end
            
            
        solver = "l2"
            

        global mse = 0
        global N_mean = 0
        for i =1:num
            
            #dist = Uniform(-1,1)
            global shr = anova_RFF.RFF_model(X_train,complex.(Y_train),U, N, "exp", dist )
            if algo == "ASHRIMP"
                UU = anova_RFF.ANOVA_boosting(shr,q,N, dependence = true, anova_step = anova_step, epsilon = epsilon, l_w = l_w, dist = dist, verbose = true)
                println("found the ANOVA index set U: ", UU)
                UU = anova_RFF.anti_downward_closed(UU)
                println("anti dwc U: ", UU)
                shr = anova_RFF.shrimp(X_train,Complex.(Y_train), UU, N, l= l,dist = dist, verbose = false)
            elseif algo == "SHRIMP"
                shr = anova_RFF.shrimp(X_train, complex.(Y_train),U ,N; prune = 0.2, l = l, 
                dist = dist , verbose = false, solver = solver )  
            elseif algo == "HARFE"
                shr = anova_RFF.harfe(X_train, complex.(Y_train),U ,N, lam = l_harfe, s = s, dist =dist)
                #anova_RFF.harfe(shr,q, N, lam = l_harfe, verbose = false, anova = false, s = s)  
            elseif algo == "AHARFE"
                UU = anova_RFF.ANOVA_boosting(shr,q,N, dependence = true, anova_step = anova_step, epsilon = epsilon, l_w = l_w, dist = dist, verbose = true)
                gsis = anova_RFF.get_gsi_w(shr)
                println("gsis:", gsis)
                println("found the ANOVA index set U: ", UU)
                UU = anova_RFF.anti_downward_closed(UU)
                println("anti dwc U: ", UU)
                if dataset == "propulsion"
                    N2 = 3000
                elseif dataset == "galaxy"
                    N2 = 10000
                elseif dataset == "airfoil"
                    N2 = 80000
                elseif dataset == "forestfire"
                    N2 = 4220
                elseif dataset == "housing"
                    N2 = 10000
                elseif dataset == "protein"
                    N2 = 4000
                elseif dataset == "elevators"
                    N2 = Int(floor(M/5 ))     
                end
                
                shr = anova_RFF.harfe(X_train, complex.(Y_train),UU ,N2, lam = l_harfe, s = s, dist = dist)
                #shr = anova_RFF.harfe(X_train,Complex.(Y_train), UU, N, l= l,dist = dd, verbose = false) 
            end

            #println("before mse: shr.U", shr.U)
            global mse += anova_RFF.get_mse(shr, X_test,Y_test)/num
            global N_mean += shr.N_best/num
            println("mse = ",anova_RFF.get_mse(shr, X_test,Y_test))
            #println("shr.U:", shr.U)
            global Y_eval = anova_RFF.evaluate(shr,X_test)
           

        end
        println("mean mse = ", mse)



    
        if num ==10
            if algo == "SHRIMP" || algo == "ASHRIMP"
                CSV.write("results.csv",DataFrame(algo  = algo, dataset = dataset, q = q, N = Int(round(N_mean)), lambda = l, epsilon = epsilon, mse = mse, N_start = N, l_w =l_w ),  append=true)
            else 
                CSV.write("results.csv",DataFrame(algo  = algo, dataset = dataset, q = q, N =s, lambda = l_harfe, epsilon = epsilon, mse = mse, N_start = N, l_w =l_w ),  append=true)
            end
        end



    #end  #q
#end   #dataset

plot(Y_test, real.(Y_eval),line=:scatter)







#=


## sorting gsis:
dist =  Normal(0,1/q)
l_w = 100
epsilon = 0.01
shr = shrimp2.shrimp_model(X_train,complex.(Y_train),U, N, "exp" , dist)
shrimp2.init_w(shr, l = l_w, dist = dist )
gsis = shrimp2.get_gsi_w(shr)
sort(collect(gsis),by = x->x[2],rev=true)
plot(sort(collect(values(gsis))) , yscale=:log10)
Y_eval = shrimp2.evaluate(shr,X_test)
plot(Y_test, real.(Y_eval), line=:scatter)

# first ANOVA cutting
qq = q
U_new = Vector{Vector{Int}}(undef, 1)
U_new[1] = []
for u in shr.U[2:end]
    if gsis[u] > epsilon || length(u)<qq
        append!(U_new,[u])
    end 
end
shr.W = Dict()
shr.U = U_new
n = Int(ceil((shr.N-1)/(length(U_new)-1)))
shr.W, ~ = shrimp2.make_W(d, qq-1, n; dist = dist, num_supports = U_new )
shr.trafo = shrimp2.make_A(shr.X, shr.W, U_new ; bas = "exp")
shr.N = n * (length(U_new)-1)+1

# second ANOVA cutting
qq = q-1
shrimp2.init_w(shr, l = l_w, dist =dist)
gsis = shrimp2.get_gsi_w(shr)
U_new = Vector{Vector{Int}}(undef, 1)
U_new[1] = []
for u in shr.U[2:end]
    if gsis[u] > 0.01 || length(u)<qq
        append!(U_new,[u])
    end 
end
shr.W = Dict()
shr.U = U_new
n = Int(ceil((shr.N-1)/(length(U_new)-1)))
shr.W, ~ = shrimp2.make_W(d, qq-1, n; dist = dist, num_supports = U_new )
shr.trafo = shrimp2.make_A(shr.X, shr.W, U_new ; bas = "exp")
shr.N = n * (length(U_new)-1)+1

# anti downward closed set:
U_new = shr.U
U_new = sort(U_new, by = length)
UU = Vector{Vector{Int64}}([])
while !isempty(U_new)
    i = popfirst!(U_new);
    if  any([issubset(i,u) for u in U_new])
    else push!(UU, i);
    end
end
X = shr.X
y = shr.y
shr.U = UU
n = Int(ceil(N/(length(UU)) ))
W, ~ = shrimp2.make_W(d, q, n; dist = dist, num_supports = UU )
trafo = shrimp2.make_A(X, W, UU ; bas = "exp")
shr.w =Dict()
shr.trafo = trafo
shr.W = W
shr.N = n* (length(UU)) 

shrimp2.approximate(shr, q, shr.N,dist = dist , prune = 0.15,l = l, verbose = false, anova = false, solver = solver)  
#shrimp2.harfe(shr, anova = false) 
mse = shrimp2.get_mse(shr, X_test,Y_test)
Y_eval = shrimp2.evaluate(shr,X_test)
plot(Y_test, real.(Y_test), line=:scatter)

=#

