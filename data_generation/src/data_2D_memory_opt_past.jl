##
## This file is for computing Saturation, 
## eigenvector of FIM and vJp for 2D Permeability.
##

using Pkg; Pkg.activate(".")

nthreads = try
    # Slurm
    parse(Int, ENV["SLURM_CPUS_ON_NODE"])
catch e
    # Desktop
    Sys.CPU_THREADS
end
using LinearAlgebra
BLAS.set_num_threads(nthreads)
println("Number of threads:", Threads.nthreads())

using JutulDarcyRules
using Random
Random.seed!(2023)
using PyPlot
using PyCall
@pyimport matplotlib.colors as mcolors
using JLD2
using Printf
using Statistics: mean, std
using Distributions
using Zygote
using Distributed
using Interpolations

# ------------------ #
# Load Dataset       #
# ------------------ #

JLD2.@load "../../../../Diff_MultiPhysics/FNO-NF.jl/scripts/wise_perm_models_2000_new.jld2" #phi = porosity

# ------------------ #
# Setting            #
# ------------------ #

## grid size
dx = 256
dy = 256
n = (dx, 1, dy)
d = (15.0, 10.0, 15.0) # in meters

# rescale permeability
function resize_array(A, new_size)
    itp = interpolate(A, BSpline(Linear()))  # Linear interpolation
    etp = extrapolate(itp, Line())          # Extrapolation method
    xs = [range(1, stop=size(A, i), length=new_size[i]) for i in 1:ndims(A)]
    return [etp(x...) for x in Iterators.product(xs...)]
end

BroadK_rescaled = resize_array(BroadK, (size(BroadK)[1], dx, dy))

# permeability
K_all = md * BroadK_rescaled
println("K_all size: ", size(K_all))

# Define JutulModel
# phi_rescaled = resize_array(phi, (dx,dy))
# ϕ = phi_rescaled
ϕ = 0.25 * ones((dx, dy))
top_layer = 70
h = (top_layer-1) * 1.0 * d[end]  
model = jutulModel(n, d, vec(ϕ), K1to3(K_all[1,:,:]; kvoverkh=0.36), h, true)

## simulation time steppings
tstep = 100 * ones(1) #in days 1000
tot_time = sum(tstep)

## injection & production
inj_loc_idx = (130, 1 , 205)
inj_loc = inj_loc_idx .* d
irate = 9e-3 #7e-3 previously
q = jutulSource(irate, [inj_loc])
S = jutulModeling(model, tstep)

figure()
imshow(reshape(BroadK_rescaled[15, :, :], n[1], n[end])', cmap="viridis")
scatter(inj_loc_idx[1], inj_loc_idx[3], color="red")
colorbar(fraction=0.04)
title("Permeability Model")
savefig("Downsampled_Perm_15.png")
close("all")

# ------------------ #
# Setting for FIM    #
# ------------------ #

nsample = 200
nev = 40  # Number of eigenvalues and eigenvectors to compute
nt = length(tstep)
μ = 0.0   # Mean of the noise
σ = 1.0   # Standard deviation of the noise
dist = Normal(μ, σ)

# ---------------------- #
# Generate Joint Samples #
# ---------------------- #

# if nprocs() == 1
#     addprocs(8, exeflags=["--threads=2"])
# end
println("num procs: ", nprocs())

# Load packages on all workers
@everywhere using Zygote, Random, LinearAlgebra, Distributions, JutulDarcyRules 

@everywhere function compute_pullback(Fp, col_U)
    try
        println("col_U size: ", size(col_U))
        return @time Fp(col_U)[1]
    catch e
        println("Pullback computation failed: ")
        return zeros(length(col_U))  # Return zeros on failure
    finally
        GC.gc()  # Free memory on this worker after computation
    end
end

# @everywhere function compute_pullback_with_noise(j, Fp, noise, cur_state_sat)
#     println("Perturbation $(j) processed on worker $(myid())")
#     println("Size of noise", size(noise))
#     try
#         return @time Fp(noise)[1]
#     catch e
#         println("Pullback computation failed for perturbation $(j): ")
#         return zeros(length(cur_state_sat))
#     finally
#         GC.gc()
#     end
# end

# Compute pullback for a given noise perturbation (used in gradient computation)
function compute_pullback_with_noise(j, Fp, noise, cur_state_sat)
    println("Perturbation $j processed on worker $(myid()), noise size: ", size(noise))
    try
        return @time Fp(noise)[1]
    catch e
        println("Pullback computation failed for perturbation $j: ")
        return zeros(length(cur_state_sat))
    end
end

for i = 10:nsample
    Base.flush(Base.stdout)

    Ks = zeros(n[1], n[end], nsample)
    eigvec_save = zeros(n[1], n[end], nev, 5)
    one_Jvs = zeros(n[1]*n[end], nev, 5)
    conc = zeros(n[1], n[end], 1)

    println("sample $(i)")
    K = K_all[i, :, :]

    # 0. update model
    model = jutulModel(n, d, vec(ϕ), K1to3(K; kvoverkh=0.36), h, true)
    S = jutulModeling(model, tstep)

    # 1. compute forward: input = K
    mesh = CartesianMesh(model)
    logTrans(x) = log.(KtoTrans(mesh, K1to3(x)))
    state00 = jutulSimpleState(model)
    state0 = state00.state  # 7 fields
    states = []

    # Repeat for 5 time steps
    for time_step in 1:5
        println("Sample $(i) time step $(time_step)")
        state(x) = S(logTrans(x), model.ϕ, q; state0=state0, info_level=1)[1]
        state_sat(x) = Saturations(state(x)[:state])

        cur_state = state(K)

        @time cur_state_sat, Fp = Zygote.pullback(state_sat, vec(K))  # v^TJ pullback
        state0_temp = copy(cur_state[:state])
        cur_state_pressure = cur_state[:state][:Pressure]

        # if i % 10 == 0
        #     figure()
        #     imshow(reshape(cur_state_sat, n[1], n[end])', cmap="viridis")
        #     colorbar(fraction=0.04)
        #     title("Saturation at time step=$(time_step)")
        #     savefig("img_$(nev)/Sample_$(i)_Saturation_$(time_step).png")
        #     close("all")

        #     figure()
        #     imshow(reshape(cur_state_pressure, n[1], n[end])', cmap="Reds")
        #     colorbar(fraction=0.04)
        #     title("Pressure at time step=$(time_step)")
        #     savefig("img_$(nev)/Sample_$(i)_Pressure_$(time_step).png")
        #     close("all")
        # end

        push!(states, cur_state_sat)

        # ------------ #
        # Compute FIM  #
        # ------------ #
        dll = zeros(n[1]*n[end], nev)
        # noise_vectors = @time generate_orthogonal_masked_noise(cur_state_sat, size(cur_state_sat), nev)
        noise_vectors = rand(dist, (n[1]*n[end], nev))
        # noise_vectors = ones(n[1]*n[end], nev)
        num_zeros = sum(noise_vectors .== 0) 
        println("number of zeros in noise vectors", num_zeros )

        # gradient_results = pmap(j -> begin
        #     noise = noise_vectors[:,j] # indexing ... [:,j]
        #     compute_pullback_with_noise(j, Fp, noise, cur_state_sat)
        # end, 1:nev)

        # Use multi-threading for computing the gradient for each noise vector
        gradient_results = Vector{Any}(undef, nev)
        Threads.@threads for j in 1:nev
            intermediate = compute_pullback_with_noise(j, Fp, noise_vectors[:, j], cur_state_sat)
            println("size of intermediate", size(intermediate))
            gradient_results[:,j] = intermediate
        end
        dll .= gradient_results
     
        # Free noise_vectors now that they’re used
        noise_vectors = nothing

        dll .= hcat(gradient_results...)
        println("size dll", size(dll))

        @time U_svd, S_svd, VT_svd = LinearAlgebra.svd(dll)
        println("size U_svd: ", size(U_svd), " S_svd: ", size(S_svd), " VT_svd: ", size(VT_svd))
        num_zeros_U = sum(U_svd .== 0) 
        println("Number of zeros from probing vector:", num_zeros_U) #0
        eigvec_save[:, :, :, time_step] = reshape(U_svd, n[1], n[end], nev)

        if i == 1
            figure()
            semilogy(S_svd, "o-")
            xlabel("Index")
            ylabel("Singular Value")
            title("Singular Value Decay at time step = $(time_step)")
            grid(true)
            savefig("img_$(nev)/Sample_$(i)_SingularValue_$(time_step)_64.png")
            close("all")

            for j in 1:nev
                figure()
                maxabs_U = maximum(abs, U_svd[:, j])*0.8
                linthresh = 0.1 * maxabs_U
                norm_U = PyPlot.matplotlib.colors.SymLogNorm(linthresh=linthresh, vmin=-maxabs_U, vmax=maxabs_U)
                imshow(reshape(U_svd[:, j], n[1], n[end])', cmap="seismic", norm=norm_U)#norm=mcolors.CenteredNorm(0))
                colorbar(fraction=0.04)
                title("Left Singular Vector $(j) at time step = $(time_step)")
                filename = "img_$(nev)/Sample_$(i)_U_svd_$(time_step)_$(j).png"
                savefig(filename)
                close("all")
            end
        end

        println("Compute vTJ")
        Jv_results = @time pmap(e -> compute_pullback(Fp, U_svd[:, e]), 1:nev)
        Jv_matrix = hcat(Jv_results...)
        println("Jv_matrix size: ", size(Jv_matrix))
        num_zeros_Jv = sum(Jv_matrix .== 0)
        println("Number of 0s in Jv", num_zeros_Jv) #0

        if i == 1
            for j in 1:nev
                figure()
                maxabs = maximum(abs, Jv_matrix[:, j])*0.8
                linthresh_J = 0.1 * maxabs
                norm_J = PyPlot.matplotlib.colors.SymLogNorm(linthresh=linthresh_J, vmin=-maxabs, vmax=maxabs)
                imshow(reshape(Jv_matrix[:, j], n[1], n[end])', cmap="seismic", norm=norm_J)#norm=mcolors.CenteredNorm(0))
                colorbar(fraction=0.04)
                title("Jacobian Vector Products with LSV $(j) at t = $(time_step)")
                filename = "img_$(nev)/Sample_$(i)_vjp_$(time_step)_$(j).png"
                savefig(filename)
                close("all")
            end
        end

        for e in 1:nev
            one_Jvs[:, e, time_step] = Jv_matrix[:, e]
        end

        # Free large arrays after each time step
        dll = nothing
        U_svd = nothing
        S_svd = nothing
        VT_svd = nothing
        Jv_matrix = nothing
        GC.gc()

        state0 = deepcopy(state0_temp)
    end
    save_object("num_ev_$(nev)/FIM_eigvec_sample_$(i).jld2", eigvec_save)
    save_object("num_ev_$(nev)/FIM_vjp_sample_$(i).jld2", one_Jvs)
    save_object("num_ev_$(nev)/states_sample_$(i).jld2", states)

    # Clean up after each sample
    eigvec_save = nothing
    one_Jvs = nothing
    states = nothing
    GC.gc()
end
