using StatsBase
using DataFrames
using ANOVAapprox
using LinearAlgebra
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

algo = "ASHRIMP"
q = 2
dataset = "airfoil"

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


#save samples:
CSV.write("samples.csv",DataFrame(hcat(Y_train, X_train', ), :auto))    ## y-values is zeroth column

d, M = size(X_train)

N = Int(floor(5*M   ))
l = 1e-12
epsilon = 0.01
l_w = 1
l_harfe = 10^(-10)
s = 300



###########################################################################
## for first diagram with gsis:
q = 1
dist =  Normal(0,1/q)
l_w = 100
U = reduce(vcat,[collect(combinations(1:d,qq)) for qq = 0:q])
shr = anova_RFF.RFF_model(X_train,complex.(Y_train), U,N,"exp" , dist)
anova_RFF.init_w(shr, l = l_w, dist = dist )
gsis = anova_RFF.get_gsi_w(shr)
sort(collect(gsis),by = x->x[2],rev=true)
plot(sort(collect(values(gsis))) , yscale=:log10)
println("gsis for RFF:")
for i in collect(1:d)
    println("(",i,",", gsis[[i]],")")
end


#with trafo:
X_train_s = ANOVAapprox.transform_R(X_train;  sigma = false, X_test =false, KDE = "DPI")
X_test_s = ANOVAapprox.transform_R(X_train;  sigma = false , X_test = X_test, KDE = "DPI")
X_test_s[findall(x-> x<-0.5, X_test_s)] .= -0.4999999
X_test_s[findall(x-> x>=0.5, X_test_s)] .= 0.49999999
X_train_s[findall(x-> x<-0.5, X_train_s)] .= -0.4999999
X_train_s[findall(x-> x>=0.5, X_train_s)] .= 0.49999999
f = ANOVAapprox.approx( X_train_s , Y_train, 1, [3], "chui2")
ANOVAapprox.approximate( f, max_iter=200,0.0 )
gsis2 = ANOVAapprox.get_GSI(f, 0.0)
println("gsis for Trafo:")
for i in collect(1:d)
    println("(",i,",", gsis2[[i]],")")
end

###########################################################################
## for second diagram with gsis:
if dataset == "airfoil"
    X_train = X_train[1:5,:]
    X_test = X_test[1:5,:]
end
d, M = size(X_train)
q = 2
dist =  Normal(0,1/q)
l_w = 100
U = reduce(vcat,[collect(combinations(1:d,qq)) for qq = 0:q])
shr =anova_RFF.RFF_model(X_train,complex.(Y_train),U, N, "exp" , dist)
anova_RFF.init_w(shr, l = l_w, dist = dist )
gsis = anova_RFF.get_gsi_w(shr)

println("gsis for RFF:")
for i in collect(1:d)
    if in([i], shr.U)
        println("(",i,",", gsis[[i]],")")
    else
        println("(",i,",", 0,")")
    end
end
global idx = d+1
for i in collect(1:d)
    for j in collect(i+1:d)
        u =[i,j]
        if in(u, shr.U)
            println("(",idx,",", gsis[u],")")
        else
            println("(",idx,",", 0,")")
        end
        global idx +=1
    end
end

##for trafo:
X_train_s=X_train_s[1:5,:]
X_test_s=X_test_s[1:5,:]
f = ANOVAapprox.approx( X_train_s , Y_train, 2, [3,3], "chui2")
ANOVAapprox.approximate( f, max_iter=200,0.0 )
gsis2 = ANOVAapprox.get_GSI(f, 0.0, dict = true)
println("gsis for trafo:")
for i in collect(1:d)
    if in([i], f.U)
        println("(",i,",", gsis2[[i]],")")
    else
        println("(",i,",", 0,")")
    end
end
global idx = d+1
for i in collect(1:d)
    for j in collect(i+1:d)
        u =[i,j]
        if in(u, f.U)
            println("(",idx,",", gsis2[u],")")
        else
            println("(",idx,",", 0,")")
        end
        global idx += 1
    end
end




######################################################
### plot ANOVA terms


if algo == "SHRIMP" || algo == "HARFE"
    U = collect(combinations(1:d,q))
elseif algo == "ASHRIMP"|| algo == "AHARFE"
    U = reduce(vcat,[collect(combinations(1:d,qq)) for qq = 0:q])
end

dist = Normal(0,1/q)
#solver = "l2"

#shr = shrimp2.shrimp_model(X_train,complex.(Y_train),U, N, "exp", dist )
#shrimp2.init_w(shr, l = l_w, dist =dist)


# if algo == "ASHRIMP"
#     shrimp2.approximate(shr, q, N,dist = dist, prune = 0.15,l = l, verbose = false, anova = true, dependent = true, solver = solver , epsilon = epsilon, l_w = l_w)
# end

#println("mse=",shrimp2.get_mse(shr, X_test,Y_test))

U = shr.U 



y_eval = Dict()
for i in collect(1:d)
    for j in collect(i+1:d)
        u = [i,j]
        if u in shr.U
            y_eval[u] = abs.(shr.trafo[ u ]*shr.w[u]  )
            CSV.write("f$i$(j)ASHRIMP.csv",DataFrame(hcat(X_train[u,:]', y_eval[u]), :auto))

        end

    end
end




# y_eval = Dict()
# for i in collect(1:d)
#     for j in collect(i+1:d)
#         u = [i,j]
#         if u in shr.U
#             X_t = range.([minimum(X_train[i,:]),minimum(X_train[j,:])],[maximum(X_train[i,:]),maximum(X_train[j,:])],length =11)
#             X_tt = collect.(Iterators.product(collect(X_t[1]), collect(X_t[2])))
#             X_ttt = zeros(d,121)
#             X_ttt[[i,j],:] = Matrix(permutedims(hcat(vec(X_tt)...))')
#             y_eval[u] = zeros(size(X_ttt,2))
#             B = shrimp2.make_A(X_ttt, shr.W,shr.U;bas = "exp")
#             y_eval[u] =  abs.(B[u][:,shr.idx_list[shr.N_best][u]]*shr.w[u] )
            
#             #CSV.write("f$i$(j)ASHRIMP.csv",DataFrame(hcat(X_01',y_eval[u]) , :auto))
#             CSV.write("f$i$(j)ASHRIMP.csv",DataFrame(hcat(Matrix(permutedims(hcat(vec(X_tt)...))')',y_eval[u]) , :auto))

#         end

#     end
# end



################################
#plot ANOVA terms trafo

y_eval = Dict()
for i in collect(1:d)
    for j in collect(i+1:d)
        u = [i,j]
        if u in f.U
            X_t = range.([minimum(X_train_s[i,:]),minimum(X_train_s[j,:])],[maximum(X_train_s[i,:]),maximum(X_train_s[j,:])],length =15)
            X_t_s = ANOVAapprox.transform_R(X_train;  sigma = false , X_test = X_t, KDE = "DPI")
            X_tt = collect.(Iterators.product(collect(X_t_s[1]), collect(X_t_s[2])))
            X_tt_s = collect.(Iterators.product(collect(X_t[1]), collect(X_t[2])))
            X_ttt = zeros(d,225)
            X_ttt[[i,j],:] = Matrix(permutedims(hcat(vec(X_tt)...))')
            idxx = findfirst(x->x==[u],f.U)
            y_eval[u] = ANOVAapprox.evaluateANOVAterms(f,0.0)[idxx,:]
            
            CSV.write("f$i$(j)trafo.csv",DataFrame(hcat(Matrix(permutedims(hcat(vec(X_tt_s)...))')',y_eval[u]) , :auto))

        end

    end
end


 