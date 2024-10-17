using IterativeSolvers
using Random, Distributions
using Combinatorics
using SparseArrays

### struct that saves all information about a random feature model:
mutable struct RFF_model
    basis::String
    X::Matrix{Float64}
    y::Union{Vector{ComplexF64},Vector{Float64}}
    W:: Dict{Any,Any}
    U::Vector{Vector{Int}}
    N::Union{Vector{Int},Int}
    trafo::Dict{}                                       # transfomration matrix A as a Dict
    w::Union{Vector{ComplexF64},Vector{Float64},Dict{Any,Any}}
    idx_list::Union{Vector{Int},Dict{Any,Any}}          # list of chosen features for shrimp    
    N_best::Int                                         # for final approximation in shrimp            
    dist
    feature_list                                      # list for indices when features witten as matrix W 


    function RFF_model(
        X::Matrix{Float64},
        y::Union{Vector{ComplexF64},Vector{Float64}},
        U::Vector{Vector{Int}},
        N::Union{Vector{Int},Int},
        basis::String = "exp",
        dist = Normal(0,1),
    )
    
        if basis in bases
            M = size(X, 2)
            d = size(X, 1)
            q = maximum([length(u) for u in U])

            if !isa(y, vtypes[basis])
                error(
                    "complex vectors for exp basis, real vectors for cos/sin basis.",
                )
            end

            if length(y) != M
                error("y needs as many entries as X has columns.")
            end

            if (length(N) != length(U)) && (length(N) != q) && (length(N) != 1) 
                error("N needs to be an integer of have |U| or max |u| entries.")
            end
            if typeof(N) == Int             #TODO andere Möglichkeiten: N vector ... 
                if in([], values(U))
                    n = Int(ceil((N-1)/(length(U)-1)))
                    feature_list =  vcat(1, 2 .+ n .* collect(0:length(U)-1))
                else
                    n = Int(ceil(N/length(U)))
                    feature_list =  1 .+  n .* collect(0:length(U))
                end
            end
            if in([], values(U))
                N_real = n * (length(U)-1) +1
            else
                N_real = n * length(U) 
            end
            W, ~ = make_W(d, q, n; dist = dist, U = U )
            trafo = make_A(X, W, U ; bas = "exp")

            if dist == "nfft"
                N_real = size(W[U[1]])[2] * length(U)
            end

            return new(basis, X, y, W, U, N_real, trafo, Dict(),Dict(),N_real,dist, feature_list)
        else
            error("Basis not found.")
        end
        
    end
    function RFF_model(
        X::Matrix{Float64},
        y::Union{Vector{ComplexF64},Vector{Float64}},
        basis::String = "exp",
    )
        if basis in bases
            M = size(X, 2)
            d = size(X, 1)

            if !isa(y, vtypes[basis])
                error(
                    "complex vectors for exp basis, real vectors for cos/sin basis.",
                )
            end

            if length(y) != M
                error("y needs as many entries as X has columns.")
            end

            return new(basis, X, y, Dict(), [], 0, Dict(), Dict(),Dict(),0,false,[])
        else
            error("Basis not found.")
        end
    end
    
end


#### sample points
function make_X(d, M; dist=Uniform(-1,1))
    return rand(dist,d,M)
end


function make_W(d, q, n; dist = Normal(0,1), U = false, sigma = 1, s = 1 )  
    # input: 
    # d: dimension
    # q: superposition
    # n: number of random features per ANOVA term
    # U: ANOVA_idex, if false all terms of order q are used
    # sigma: variance
    # s: parameter for distribution
    if U == false
        U = collect(combinations(1:d,q))
    end
    W = Dict()
    for u in U
        if u == []                                           #TODO: bei allen anderen Dichten auch den Koeff. zu [] einzeln betrachten
            W[u] = [1]
        else
            if typeof(dist) == Float64
                W[u] = rand( TDist(2*dist-1),length(u), n) .* sqrt(2*dist-1) .* sigma
            elseif typeof(dist) == "Sobolev"
                W[u] = rand( TDist(2*s-1),length(u), n) .* sqrt(2*s-1) .* sigma
            elseif dist == "Cauchy"
                W[u] = rand(Cauchy(0,sigma),length(u),n)
            else
                W[u] = rand(dist, length(u), n)
            end
        end
    end
    return W, U
end

#### several functions for constructing the matrix/Dict A: 
function make_A(X, W, U;bas = "exp")
    # input:
    # X: samples
    # W: random features
    # U: ANOVA index-set
    # bas: basis 
    A = Dict()
    if bas == "exp"
        for u in U
            if u == []
                A[[]] = ones(size(X,2))
            else
                A[u] = exp.(im * X[u,:]'*W[u])
            end
            
        end   
    elseif bas == "tanh"
        for u in U
            if u == []
                A[[]] = ones(size(X,2))
            else
                A[u] = exp.(im * X[u,:]'*W[u])
            end
            
        end  
    end
    return A
end


function make_A_matrix(X, W,U;bas = "exp")
    A =(zeros(size(X,2)))
    if bas == "exp"
        for u in U
            if u == []
                A = hcat(A, ones(size(X,2)))
            else
                A = hcat(A, exp.(im * X[u,:]'*W[u]))
            end
            
        end   
    end 
    return A[:,2:end]
end

function make_A(shr)
    A = (zeros(size(shr.X,2)))
    if shr.basis == "exp"
        for u in shr.U
            if u == []
                A = hcat(A, ones(size(shr.X,2)))
            else
                A = hcat(A, exp.(im * shr.X[u,:]'*shr.W[u]))
            end
        end   
    end 
    return A[:,2:end]
end



function ANOVA_boosting(shr, q, N ; dependence = false, anova_step= "descent", epsilon = 0.05, l_w = 0.1, dist=Normal(0,1),verbose=false )
    # input: 
    # shr:              RFF_model
    # q:                maximal superposition
    # N:                number of random features

    # dependece of input variables: true/false
    # anova_step:       descent/ascent
    # epsilon: cut_off paramter
    # l_w:              regularization paramter
    # dist:             distribution of random features 
    # verbose:          if details should be outputed

    # 
    M = size(shr.X, 2)
    d = size(shr.X, 1)

    if anova_step == "descent"
        if dependence == true
            U = reduce(vcat,[collect(combinations(1:d,qq)) for qq = 0:q])
        elseif dependence ==false
            U = collect(combinations(1:d,q))
        end
    elseif anova_step == "ascent"
        if dependence == true
            U = reduce(vcat,[collect(combinations(1:d,qq)) for qq = 0:1])
        elseif dependence ==false
            U = collect(combinations(1:d,1))
        end
            
    end
    shr = anova_RFF.RFF_model(shr.X,shr.y,U,N, "exp",  dist)
    
    

    if dependence == true
        shr = anova_RFF.init_w(shr; l = l_w, dist =dist)
        if anova_step == "descent"
            for qq = q:-1:1
                shr = anova_RFF.init_w(shr, l = l_w, dist =dist)
                gsis = anova_RFF.get_gsi_w(shr)
                if verbose == true 
                    println("gsis:", gsis)
                end
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
    
                shr.W, ~ = anova_RFF.make_W(d, qq-1, n; dist = dist, U = U_new )
                shr.trafo = anova_RFF.make_A(shr.X, shr.W, U_new ; bas = "exp")
                shr.N = n * (length(U_new)-1)+1
    
            end
    
        elseif anova_step =="ascent" 
            for qq = 1:1:q
                shr = anova_RFF.init_w(shr, l = l_w, dist =dist)
                gsis = anova_RFF.get_gsi_w(shr)
                if verbose == true 
                    println("gsis:", gsis)
                end
                U_new = Vector{Vector{Int}}(undef, 1)
                U_new[1] = []
                for u in shr.U[2:end]
                    if gsis[u] > epsilon 
                        append!(U_new,[u])
                    end 
                end
    
                # append new u's with cardinality qq+1 
                if qq < q
                    U_new2 = copy(U_new)
                    for u in U_new
                        for v in U_new
                            if in(unique(sort(vcat(u,v))),U_new2) || length(unique(sort(vcat(u,v)))) >qq+1
                            else 
                                append!(U_new2,[sort(vcat(u,v))])
                            end
                        end
                    end
                    U_new = U_new2
                end
    
                
                shr.W = Dict()
                shr.U = U_new
                println("U_new::", U_new)
                n = Int(ceil((shr.N-1)/(length(U_new)-1)))
    
                shr.W, ~ = anova_RFF.make_W(d, qq+1, n; dist = dist, U = U_new )
                shr.trafo = anova_RFF.make_A(shr.X, shr.W, U_new ; bas = "exp")
                shr.N = n * (length(U_new)-1)+1
                #println("shr.U:",shr.U)

            end
        end
    elseif dependence ==false
        shr = anova_RFF.init_rff(shr; l = l_w, dist =dist)
        val_idx = collect(1:M)     
        if anova_step == "descent"
            for qq in 1:q
                gsis = anova_RFF.get_gsi(shr,val_idx)             
                if typeof(epsilon) == Vector{Float64}
                    useless = keys(filter(p->(last(p)  <= epsilon[length(p)] ),gsis))
                else
                    useless = keys(filter(p->(last(p)  <= epsilon ),gsis))
                end
                if verbose == true
                    println("gsis:")
                    print(gsis)
                end
                if length(useless) !=0
                    N_omit = sum([length(shr.w[u]) for u in useless] )
                end
    
                U_new = []
    
                for uu in useless 
                    if length(uu)>1
                        U_new = vcat(U_new,collect(combinations(uu,length(uu)-1)) )
                    end
                    delete!(shr.W, uu )
                end
    
                U_new = unique(U_new)     #every new u only one time   
                   
                if typeof(epsilon) == Vector{Float64}
                    shr.U = collect(keys(filter(p->(last(p)  >= epsilon[length(p)]),gsis))) 
                else
                    shr.U = collect(keys(filter(p->(last(p)  >= epsilon),gsis)))        # u's that we keep 
                end
                
                deleteat!(U_new,findall(x->in(x,shr.U),U_new))                      ### delete all u's which we already have
                if shr.U != []
                    deleteat!(U_new, findall(x-> any([issubset(x,u) for u in shr.U]),U_new))
                    
    
                end
                if length(U_new ) == 0                                       #no new indices
                else
                    shr.U = vcat(shr.U,U_new)                                           # add 
    
                end
    
                #draw new samples on every u:
                n = Int(ceil(shr.N/(length(shr.U))))
                shr.W, ~ = anova_RFF.make_W(d,q, n; dist =shr.dist, U =shr.U )
                shr.trafo = anova_RFF.make_A(shr.X, shr.W, shr.U ; bas = "exp")
                shr.N = n*length(shr.U)
                shr.N_best = n*length(shr.U)
    
                A = shr.trafo[shr.U[1]]
                for inds in shr.U[2:end]
                    A = hcat(A,shr.trafo[inds])
                end
                w = A'*(A*A'/M+l_w*I(length(shr.y)))^(-1)*shr.y/M
    
                shr.w = Dict()
                ind = 1
                for i in 1:length(shr.U)                       
                    shr.w[shr.U[i]] = w[collect(ind : ind +size(shr.W[shr.U[i]],2)-1)  ]
                    ind += length(shr.w[shr.U[i]])
                end
    
            end
        elseif anova_step == "ascent"
    
            for qq in 1:1:q
                gsis = anova_RFF.get_gsi(shr,val_idx)             
                if verbose == true
                    println("gsis:", gsis)
                end
                U_new = Vector{Vector{Int}}(undef, 1)
                U_new[1] = []
                if typeof(epsilon) == Vector{Float64}
                    for u in shr.U[1:end]
                        if  length(u) >0
                            if gsis[u] > epsilon[length([u])]
                                append!(U_new,[u])
                            end 
                        end
                    end
                else
                    for u in shr.U[1:end]
                        if  length(u) >0
                            if gsis[u] > epsilon
                                append!(U_new,[u])
                            end 
                        end
                    end
                end
                if verbose == true
                    println("new U: ", U_new)
                end
                if qq < q
                    U_new2 = copy(U_new)
                    for u in U_new
                        for v in U_new
                            if in(unique(sort(vcat(u,v))),U_new2) || length(unique(sort(vcat(u,v)))) >qq+1
                            else 
                                append!(U_new2,[sort(vcat(u,v))])
                            end
                        end
                    end
                    U_new = U_new2
                end
                if verbose == true
                    println("new U:", U_new)
                end
                
                shr.W = Dict()
                shr.U = U_new
                n = Int(ceil((shr.N-1)/(length(U_new)-1)))
    
                shr.W, ~ = anova_RFF.make_W(d, qq+1, n; dist = dist, U = U_new )
                shr.trafo = anova_RFF.make_A(shr.X, shr.W, U_new ; bas = "exp")
                shr.N = n * (length(U_new)-1)+1
    
    
                A = shr.trafo[shr.U[1]]
                for inds in shr.U[2:end]
                    A = hcat(A,shr.trafo[inds])
                end
    
                w = A'*(A*A'/M+l_w*I(length(shr.y)))^(-1)*shr.y/M
    
                shr.w = Dict()
                ind = 1
                for i in 1:length(shr.U)                       
                    #shr.w[shr.U[i]] = w[(i-1)* n + 1: i*n ]
                    shr.w[shr.U[i]] = w[collect(ind : ind +size(shr.W[shr.U[i]],2)-1)  ]
                    ind += length(shr.w[shr.U[i]])
                end
    
            end
    
        end


    end


    #finally return set U

    return shr.U
end


## function which does the first approximation with regularization:
function init_w(shr; l = 1e-3, dist =Normal(0,1))
    N = shr.N
    M = size(shr.X, 2)
    d = size(shr.X, 1)
   
    U = shr.U 
    W_reg = spzeros(Complex{Float64},shr.N,shr.N)
    if U[1] == [] 
        ind = 2  # if [] is included 
        i_beg = 2
    else
        ind = 1
        i_beg = 1
    end
    for i in i_beg:length(U)
        u = U[i]
        n = size(shr.W[u],2)
        if length(u) == 1

            W_reg[collect(ind : ind +n-1), collect(ind : ind +n-1)] = shr.trafo[u]'*ones(M,M) * shr.trafo[u] ./M^2
        elseif length(u) >1
            u_sub = collect(powerset(u)) 
            pop!(u_sub)
            W_u = zeros(M,M)
            for uu in u_sub
                if in(uu, values(U)) 
                W_u = W_u + shr.trafo[uu] * shr.trafo[uu]'
                end
            end

            W_reg[collect(ind : ind +n-1), collect(ind : ind +n-1) ] = sqrt(Matrix(shr.trafo[u]'*W_u * shr.trafo[u] ./M^2))
        end
        ind += n

    end
    A = shr.trafo[shr.U[1]]
    for inds in shr.U[2:end]
        A = hcat(A,shr.trafo[inds])
    end
    ww = lsqr(vcat(hcat(A), sqrt(l) .* W_reg), vcat( shr.y,zeros(shr.N)) )
    shr.w = Dict()
    ind = 1
    for i in 1:length(shr.U)                       
        shr.w[shr.U[i]] = ww[collect(ind : ind +size(shr.W[shr.U[i]],2)-1)  ]
        ind += length(shr.w[shr.U[i]])
    end


    return shr
end

## function which does the first approximation:
function init_rff(shr; q= 1, l = 1e-6, dist = Normal(0,1), solver ="l2", maxiter = 100)

    n = Int(floor(shr.N/length(shr.U)))
    N_real = shr.N 

    if shr.W == []
        shr.W, ~ = anova_RFF.make_W(d, q, n; dist =dist, U = shr.U )
        shr.trafo = anova_RFF.make_A(shr.X, shr.W, shr.U ; bas = "exp")
    end
    shr.idx_list[N_real] = Dict()
    for u in shr.U 
        shr.idx_list[N_real][u] = 1:n
    end

    A = shr.trafo[shr.U[1]]
    for u in shr.U[2:end]
        A = hcat(A,shr.trafo[u])
    end



    if solver == "lsqr"
        w = lsqr(A, shr.y, maxiter = maxiter, damp = l,atol = 1e-12)
    elseif solver == "l2"
        M = size(shr.X, 2)
        w = A'*(A*A'/M+l*I(length(shr.y)))^(-1)*shr.y/M
    end
    shr.w = Dict()
    if shr.U[1] == []
        shr.w[[]] = [w[1]]
        for i in 2:length(shr.U)                        
            shr.w[shr.U[i]] = w[(i-2)* n + 2: (i-1)*n+1 ]
        end
    else
        for i in 1:length(shr.U)                        
            shr.w[shr.U[i]] = w[(i-1)* n + 1: i*n ]
        end
    end
    
    return shr
end


### evaluate at points X:
function evaluate(
    shr;
    )::Union{Vector{ComplexF64},Vector{Float64}}
    N = shr.N_best
    y_eval = zeros(size(shr.X,2))
    for u in shr.U
        if u == []
            y_eval = y_eval .+ shr.w[u] 
        else
            y_eval = y_eval + shr.trafo[u][:,shr.idx_list[N][u]]*shr.w[u] 
        end
    end
    return y_eval
end

### evaluate at other points:
function evaluate(
    shr,
    X::Matrix{Float64}
    )::Union{Vector{ComplexF64},Vector{Float64}}
    N = shr.N_best
    B = anova_RFF.make_A(X, shr.W,shr.U;bas = "exp")
    y_eval = zeros(size(X,2))
    for u in shr.U
        if length(shr.w[u])!=0
            if u == [] 
                y_eval = y_eval .+ shr.w[u] 
            else
                y_eval = y_eval + B[u][:,shr.idx_list[N][u]]*shr.w[u] 
            end
        end
    end

    return y_eval
end

### calculate the mean square error:
function get_mse(shr, X_test, y_test)
    y_pred = evaluate(shr, X_test)
    return 1 / length(y_test) * (norm(y_pred - y_test)^2)
end

### calculate the Sobol indices for dependent input variables
function get_gsi_w(shr)
    gsis = Dict()
    M = length(shr.y)
    for u in shr.U
        if u != []
        gsis[u] = var(shr.trafo[u] * shr.w[u])/ var(shr.y, corrected=true)  
        end
    end
    #ss = sum(values(gsis))  #normalization of gsis
    #map!(x-> x/ss, values(gsis))
return gsis
end

###  calculate the Sobol indices for independent input variables
function get_gsi(shr,val_idx)::Dict
    #ind_temp = 1
    gsis = Dict()
    M = length(val_idx)
    for u in shr.U
        W = shr.W[u]
        n = length(shr.w[u])
        y_pred = shr.trafo[u][val_idx,:] * shr.w[u]
        if length(u) == 3
            A1 = exp.(im .* shr.X[u[1],val_idx] * W[1,:]')
            A2 = exp.(im .* shr.X[u[2],val_idx] * W[2,:]')
            A3 = exp.(im .* shr.X[u[3],val_idx] * W[3,:]')
            B = ([A1[i,j] for i in 1:M, j in 1:n, k in 1:M] - [A1[i,j] for k in 1:M, j in 1:n, i in 1:M]) .* ([A2[i,j] for i in 1:M, j in 1:n, k in 1:M] - [A2[i,j] for k in 1:M, j in 1:n, i in 1:M]) .* ([A3[i,j] for i in 1:M, j in 1:n, k in 1:M] - [A3[i,j] for k in 1:M, j in 1:n, i in 1:M])

            gsis[u] = var(1/M .* sum(B,dims=1)[1,:,:]' * shr.w[u], corrected = true)

        elseif length(u) == 2

            A1 = exp.(im .* shr.X[u[1],val_idx] * W[1,:]')
            A2 = exp.(im .* shr.X[u[2],val_idx] * W[2,:]')

            B = ([A1[i,j] for i in 1:M, j in 1:n, k in 1:M] - [A1[i,j] for k in 1:M, j in 1:n, i in 1:M]) .* ([A2[i,j] for i in 1:M, j in 1:n, k in 1:M] - [A2[i,j] for k in 1:M, j in 1:n, i in 1:M])

            gsis[u] = var(1/M .* sum(B,dims=1)[1,:,:]' * shr.w[u], corrected = true)
            

        
        elseif length(u) == 1
            gsis[u] = var( y_pred, corrected = true)                       
        elseif length(u) == 4
            A1 = exp.(im .* shr.X[u[1],val_idx] * W[1,:]')
            A2 = exp.(im .* shr.X[u[2],val_idx] * W[2,:]')
            A3 = exp.(im .* shr.X[u[3],val_idx] * W[3,:]')
            A4 = exp.(im .* shr.X[u[4],val_idx] * W[4,:]')
            B = ([A1[i,j] for i in 1:M, j in 1:n, k in 1:M] - [A1[i,j] for k in 1:M, j in 1:n, i in 1:M]) .* ([A2[i,j] for i in 1:M, j in 1:n, k in 1:M] - [A2[i,j] for k in 1:M, j in 1:n, i in 1:M]) .* ([A3[i,j] for i in 1:M, j in 1:n, k in 1:M] - [A3[i,j] for k in 1:M, j in 1:n, i in 1:M]) .* ([A4[i,j] for i in 1:M, j in 1:n, k in 1:M] - [A4[i,j] for k in 1:M, j in 1:n, i in 1:M])

            gsis[u] = var(1/M .* sum(B,dims=1)[1,:,:]' * shr.w[u], corrected = true)

        end
        #ind_temp = ind_temp+n
    end
    #ss = 1   
    #ss = var(shr.y, corrected=true)
    ss = sum(values(gsis))  
    map!(x-> x/ss, values(gsis))   #normalize
    return gsis

end

### auxiliary function 
function anti_downward_closed(U)
    to_delete = []
    for u in U 
        if u == []
            to_delete = vcat(to_delete, findall(x->x == u, U))
        else
            if sum(issubset.([u],U))>1 
                
                to_delete = vcat(to_delete, findall(x->x == u, U))
            end
        end
    end
    deleteat!(U,to_delete)
    return U
end

### realizing shrimp algorithm:
function shrimp(X,y, U, N; numCV::Int =10, steps::Int =100, prune::Float64 = 0.25, verbose::Bool =true, l = 1e-6, dist = Normal(0,1), solver = "lsqr", maxiter = 100 )
    train_mse = Dict()
    val_mse = Dict()
    ind_list = Dict()

    M = size(X, 2)
    d = size(X, 1)
    cvIter = 1
    val_idx = Int.(collect((cvIter-1)*M/numCV+1:cvIter*M/numCV))
    train_idx =  setdiff(1:M,val_idx)
    ytr = y[train_idx]
    yval = y[val_idx]

    shr = anova_RFF.RFF_model(X,y,U,N, "exp",  dist)        ## TODO: different dist for different ANOVA terms (adapt variance to q?)

    shr = anova_RFF.init_rff(shr, l=l, dist = dist, solver = solver)

    w_u = copy(shr.w)
    w_prune = copy(w_u[shr.U[1]])
    for i in 2:length(shr.U)
        append!(w_prune,copy(w_u[shr.U[i]]))
    end

    ind_list[length(w_prune)] = Dict()
    for u in shr.U 
        ind_list[length(w_prune)][u] = 1:length(shr.w[u])
    end

    A = shr.trafo[shr.U[1]]
    for inds in shr.U[2:end]
        A = hcat(A,shr.trafo[inds])
    end
    Atr = A[train_idx, :]
    Aval = A[val_idx, :]

    for i in 1:steps   
        
        N_old =  length(w_prune)
        if N_old <= 2
            break
        end
        thres = quantile(abs.(w_prune),prune)
        idx = findall(abs.(w_prune) .>=thres)
        idx_save = copy(idx)
        Atr = Atr[:, idx]
        Aval = Aval[:, idx]
        N_new = length(idx)
        ind_list[N_new] = Dict()
        for u in shr.U
            n = length(shr.w[u])
            if n!=0
                temp = findlast(x->x<=n,idx)
                if temp == nothing
                    ind_list[N_new][u] = []
                    idx .-= n 
                    
                else
                    idx_u = collect(1:temp)
                    ind_list[N_new][u] = ind_list[N_old][u][idx[idx_u]]
                    deleteat!(idx,1:length(idx_u))
                    idx .-= n    
                end
            else
                ind_list[N_new][u] = []
            end
        end
        if solver == "lsqr"
            w_prune = lsqr(Atr, ytr, maxiter =maxiter, damp = l, atol = 1e-12)

        elseif solver == "l2"
            if M>=N_new
                w_prune = (Atr'*Atr/M+l*I(N_new))^(-1)*Atr'* ytr/M 
            else
                w_prune = Atr'*(Atr*Atr'/M+l*I(length(ytr)))^(-1)*ytr/M
            end
        end 

        ### save w[u] 
        w_prune_c = copy(w_prune)
        for u in shr.U
            n = length(shr.w[u])
            if n != 0
                temp = findlast(x->x<=n,idx_save)
                if temp == nothing
                    shr.w[u] = []
                    idx_save .-= n
                else
                    idx_u = collect(1:temp)
                    shr.w[u] = w_prune_c[idx_u]
                    deleteat!(idx_save,1:length(idx_u))
                    deleteat!(w_prune_c,1:length(idx_u))
                    idx_save .-= n                 
                end
            end
        end
        

        # collect errors:

        train_mse[N_new] = sum(abs.(ytr-Atr * w_prune).^2)/(length(ytr))
        val_mse[N_new] = sum(abs.(yval- Aval * w_prune ).^2)/(length(ytr))
        if verbose == true 
            println("N = ",N_new)
            println("Train mse: ", train_mse[N_new])
            println("Val mse: ", val_mse[N_new])
            println("---------------------------")
        end


    end


    ### final approx:
    min_mses, N_best = findmin(val_mse)
    shr.N_best =  N_best
    Atr = zeros(length(train_idx),1)
    Aval = zeros(length(val_idx),1)

    for u in shr.U
        if in(u, keys(ind_list[N_best]))
            Atr = hcat(Atr,shr.trafo[u][train_idx,ind_list[N_best][u]])
            Aval = hcat(Aval,shr.trafo[u][val_idx,ind_list[N_best][u]])
        end
    end
    Atr = Atr[:,2:end]
    Aval = Aval[:,2:end]


    ytr = shr.y[train_idx]
    yval = shr.y[val_idx]

    if solver == "lsqr"
        w = lsqr(Atr, ytr, maxiter =maxiter, damp = l,atol = 1e-12)
    elseif solver == "l2"
        if M>N_best
            w = (Atr'*Atr/M+l*I(N_best))^(-1)*Atr' * ytr/M
        else
            w = Atr'*(Atr*Atr'/M+l*I(length(ytr)))^(-1)*ytr/M
        end
    end 
    train_mse[N_best] = sum(abs.(ytr-Atr * w).^2)/(length(ytr))
    val_mse[N_best] = sum(abs.(yval- Aval * w ).^2)/(length(ytr))


    ##save all stuff in shr:
    shr.idx_list = ind_list
    for u in shr.U
        if in(u,keys(ind_list[N_best]))
            n = length(ind_list[N_best][u])
            shr.w[u] = w[1:n]
            deleteat!(w,1:n)
        else
            shr.w[u] = []
        end
    end

    return shr
end


function harfe( X, y ,U, N; numCV = 10, c0 =[] ,s = 30 ,mu=0.1 ,lam = 0.001,tot_iter= 200 ,tol_error = 5e-3 , tol_coeff = 1e-5, verbose =true , dist = Normal(0,1), solver = "l2")
    train_mse = Dict()
    val_mse = Dict()
    ind_list = Dict()


    M = size(X, 2)
    d = size(X, 1)
    cvIter = 1
    val_idx = Int.(collect((cvIter-1)*M/numCV+1:cvIter*M/numCV))
    train_idx =  setdiff(1:M,val_idx)
    ytr = y[train_idx]
    yval = y[val_idx]

    shr = anova_RFF.RFF_model(X,y,U,N, "exp",  dist)       

    N = shr.N 
    # Default parameter Handeling    
    c0 = zeros(N)
    error = zeros(tot_iter)
    C = Complex.(zeros(N,tot_iter+1))
    C[:,1] = c0
    i = 1
    iter_req = i
    A = make_A_matrix(X, shr.W, U;bas = "exp")

    #nn = norm(A)     
    #A = A./nn               #normalization (is done in HARFE)   

    z1 = A'*y
    z2 = A'*A
    rel_err = norm(A * c0 -y)/norm(y)
    while rel_err > 1e-10
       c_tilde = C[:,i] + mu*(z1 - z2*C[:,i]) - M* ((mu*lam)*C[:,i])
       c_tilde_sort = sort(abs.(c_tilde))
       idx = findall(x-> x.>= c_tilde_sort[shr.N-s+1],abs.(c_tilde))
       A_pruned = A[:,idx]
       z1_pruned = A_pruned'*y
       z2_pruned = A_pruned'*A_pruned
       c_pruned = pinv((z2_pruned + lam*I(length(idx)))) * z1_pruned 
       C[idx,i+1] = c_pruned
       error[i+1] = norm(A*C[:,i+1]-y)/norm(y)
       iter_req = i+1
 
       if error[i+1]<=tol_error
          break
       elseif norm(C[:,i+1]-C[:,i]) <= tol_coeff
          break
       end   
       rel_err = error[i+1]
       i = i+1
 
       if i+1==tot_iter
          iter_req = i
          break 
       end
    end
    
    ww = C[:,iter_req]
    shr.w = Dict()
    ind = 1
    for i in 1:length(shr.U)                       
        shr.w[shr.U[i]] = ww[collect(ind : ind +size(shr.W[shr.U[i]],2)-1)  ] 
        ind += length(shr.w[shr.U[i]])
    end
    shr.idx_list[shr.N] = Dict()
    for u in shr.U 
        shr.idx_list[shr.N][u] = 1:length(shr.w[u])
    end
    shr.N_best = shr.N
   
    return shr
 

    
end

### realizing mcmc
function mcmc(X,y, U, N; steps::Int =10, ga::Float64 = 3.0, delta::Float64 = 0.5, l = 1e-6, dist = Normal(0,1), solver = "lsqr", maxiter = 100 )
    M = size(X, 2)
    d = size(X, 1)
    cvIter = 1
    shr = anova_RFF.RFF_model(X,y,U,N, "exp",  dist)  

    A = anova_RFF.make_A_matrix(X, shr.W, U;bas = "exp")

    if solver == "lsqr"
        w = lsqr(A, y, maxiter =maxiter, damp = l,atol = 1e-12)
    elseif solver == "l2"
        if M>N
            w = (A'*A/M+l*I(N))^(-1)*A' * y/M
        else
            w = A'*(A*A'/M+l*I(length(y)))^(-1)*y/M
        end
    end 

    W = anova_RFF.get_matrix_W(shr)
    mask = W .!= 0 
    for i =1:steps
        W2 = copy(W)
        r = rand(Normal(0,1), size(W))  .* mask 
        W2 = W2 + delta .* r
    
        A2 = anova_RFF.make_A_from_W(X,W2,shr)

        if solver == "lsqr"
            w2 = lsqr(A2, y, maxiter =maxiter, damp = l,atol = 1e-12)
        elseif solver == "l2"
            if M>N
                w2 = (A2'*A2/M+l*I(N))^(-1)*A2' * y/M
            else
                w2 = A2'*(A2*A2'/M+l*I(length(y)))^(-1)*y/M
            end
        end  

        r = rand(N)
        indices_to_change = findall((abs.(w2) ./ abs.(w)).^ga .> r) 
        W[:,indices_to_change] = W2[:,indices_to_change]
    
        #println(length(indices_to_change))
        A[:,indices_to_change] = A2[:,indices_to_change]

        if solver == "lsqr"
            w = lsqr(A, y, maxiter =maxiter, damp = l,atol = 1e-12)
        elseif solver == "l2"
            if M>N
                w = (A'*A/M+l*I(N))^(-1)*A' * y/M
            else
                w = A'*(A*A'/M+l*I(length(y)))^(-1)*y/M
            end
        end 

    
    end
    ### save stuff in shr:
        shr.N_best = length(w)
        n = Int(length(w)/length(U ))
        shr.idx_list[shr.N_best] = Dict()
        for idx in 1:length(U )
            shr.W[U[idx]] = W[1:length(U[idx]), n*(idx-1) + 1:n*(idx)  ]
            shr.w[U[idx]] = w[ n*(idx-1) + 1:n*(idx)  ]
            shr.trafo[U[idx]] = A[:,n*(idx-1) + 1:n*(idx)]
            shr.idx_list[shr.N_best][U[idx]] = 1:n
        end
       

    return shr


end


####### for mcmc:
### function that turns Dict of random features into matrix:
function get_matrix_W(shr)
    q = maximum(length.(shr.U))
    W = zeros( q, shr.N)
    global idx = 1
    for u in shr.U 
       
        if length(u) == 0
            W[1, idx] = 1.0
            global idx = idx+ 1
        else
            n = size(shr.W[u])[2]
            W[1:length(u), idx:idx+n-1] = shr.W[u]
            global idx = idx+ n
        end
    end
    return W
end

### function that constructs matrix A from random features in W:
function make_A_from_W(X,W,shr)
    A =(zeros(size(shr.X,2)))
    if shr.basis == "exp"
        for idx in 1:length(shr.U)
            if shr.U[idx] == []
                A = hcat(A, ones(size(shr.X,2)))
            else
                A = hcat(A, exp.(im * shr.X[shr.U[idx],:]'*W[1:length(shr.U[idx]),
                shr.feature_list[idx]:shr.feature_list[idx+1] -1 ]))
            end
        end   
    end 
    return A[:,2:end] 
end