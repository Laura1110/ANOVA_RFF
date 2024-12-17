using Distributions
using Combinatorics
using IterativeSolvers
using SpecialFunctions
using LinearAlgebra
using Plots 
using CSV 
using DataFrames
using Copulas


include("../anova_RFF.jl")
function fun(X,f)   
    if f ==1
        return X[4,:].^2 .+  X[3,:] .* X[2,:] .+  X[1,:] .* X[2,:] .+  X[4,:]      #fT1
    elseif f ==2
        	return sin.(X[1,:]) + 7 .* sin.(X[2,:]).^2 + 0.1.* X[3,:].^4 .*  sin.(X[1,:]) #fT2
    elseif f ==3
        return 10 .* sin.(pi .*  X[1,:] .* X[2,:]) + 20 .* (X[3,:]  .- 0.5).^2 + 10 .* X[4,:]  + 5 .*  X[5,:]
    elseif f == 4
        return X[4,:] .+  X[3,:] .+  X[2,:] .+  X[1,:]    
    end
    

end


 

setting = 5
anz_mse = 10
epsilon = 0.05

#c =2 
for c in 3


 if setting == 1
    d = 10
    q = 2
    M = 500
    f = 2
    U_echt  = [[1,3], [2]]
    if c == 3
        theta = 5
    else
        theta = 2
    end
    if c == 2
        l_w = 100
    else
        l_w = 200
    end
elseif setting == 2
    d = 5
    q = 2
    M = 500
    f = 2
    U_echt  = [[1,3], [2]]
    if c == 3
        theta = 5
    else
        theta = 2
    end
    if c == 2
        l_w = 100
    else
        l_w = 200
    end
elseif setting == 3
    d = 20
    q = 2
    M = 500
    f = 3
    U_echt =[[1,2],[3], [4], [5]]
    l_w = 100
    theta = 3
    epsilon =0.01
elseif setting == 4
    d = 10
    q = 3
    M = 200
    f = 3
    U_echt  =[[1,2],[3], [4], [5]]
    if c ==1
        theta = 5
    elseif c == 2
        theta = 2
    elseif c == 3
        theta = 4
    end
    l_w =100
    epsilon = 0.01
elseif setting == 5
    d = 10
    q = 3
    #q = 2
    M = 500
    f = 1
    U_echt  =[[1,2],[2,3], [4]]
    if c ==1 
        theta = 3
    elseif c ==2
        theta = 2
    elseif c == 3
        theta = 4
    end
    l_w =100
    epsilon = 0.01
elseif setting ==6
    d = 20
    q = 2
    M = 500
    f = 1
    U_echt  =[[1,2],[2,3], [4]]
    if c ==1 
        theta = 2
    elseif c ==2
        theta = 1
    elseif c == 3
        theta = 4
    end
    l_w =100
    epsilon = 0.01
end





if f == 2
    X1 = Uniform(-pi,pi)
elseif f == 3
    X1 = Uniform(0,1)
elseif f == 1
    X1 = Normal(0,1)
end
if c == 1
    C = ClaytonCopula(d,theta)
elseif c== 2
    C = GumbelCopula(d,theta)
elseif c== 3
    C = FrankCopula(d,theta)
end

D = SklarDist(C,(X1 for i in 1:d))

X = rand(D,M)
X_test = rand(D,M)



### some approximation:
y = Complex.(fun(X,f)) #+ rand(Normal(0,1),M)
y_test = Complex.(fun(X_test,f))


dd = Normal(0,1/q)
dist = dd  # distribution of random features


solver = "l2"
l = 1e-12

global mse_s =zeros(anz_mse)
global mse_as = zeros(anz_mse)
global mse_es = zeros(anz_mse)


N =5*M

#kk=1
for kk =1:anz_mse
    U = collect(combinations(1:d,q)) 

    shr = anova_RFF.RFF_model(X,y,U,N, "exp",  dist)
    println("start approx 1")
    shrimp2.shrimp(X,y,U, N; numCV =10, steps =100, prune = 0.2, verbose =false, l = l, dist = dd, solver = solver)
    global mse_s[kk] = shrimp2.get_mse(shr, X_test,y_test)
    println("mse shrimp: ",mse_s[kk])


    U = reduce(vcat,[collect(combinations(1:d,qq)) for qq = 0:q])
    shr2 = anova_RFF.RFF_model(X,y, "exp",  dist)
    println("start approx 2")
    UU = anova_RFF.ANOVA_boosting(shr,q,N, dependence = true, anova_step = ascent, epsilon = epsilon, l_w = l_w, dist = dd, verbose = true)
    shr = anova_RFF.shrimp(X,y, UU, N, l= l,dist = dd, verbose = false, prune = 0.2, solver =solver)
    global mse_as[kk] = shrimp2.get_mse(shr2, X_test,y_test)
    println("mse with anova: ",mse_as[kk])
    println("U: ", shr2.U)



end  #kk

#CSV.write("numerik_dep.csv",DataFrame(d = d, q = q, M = M ,c= c,theta = theta , f=f, l=l_w, s = mean(mse_s) ,as = mean(mse_as)),  append=true)

end  #c


