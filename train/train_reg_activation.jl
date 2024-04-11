using BinaryNeuralNetwork: RegularizedLayer, activation_regulizer
using BinaryNeuralNetwork: convert2binary_weights, convert2binary_activation
using Flux: Chain, AdaBelief, mse

include("utils.jl")

# get data
train_data, test_data = createloader(batchsize=256)

# init model
# insert identity as weight regulizer
model = Chain(
    RegularizedLayer(28^2, 256, tanh, identity),
    RegularizedLayer(256, 10, tanh, identity)
)


# default accuracy ≈ 0.1
accuracy(train_data, model)
accuracy(test_data, model)

# check default binary model accuracy
binact_model = convert2binary_activation(model)
accuracy(train_data, binact_model)


function objective(m, x, y, λ2::Float64=0.00000001)
    logitcrossentropy(m(x), y) + λ2 * activation_regulizer(m, x)
end

function objective(m, x, y)
    logitcrossentropy(m(x), y)
end


function train_activation_epochs(model, opt, train, loss; history = [],
    epochs::Int=25, periods::Int=4, λmax=10e-3, λmin=10e-6)
    
    p = Progress(periods*epochs, 1)
    ps = params(model)
    # println(model)
    binact_model = convert2binary_activation(model)
    
    ar = activation_regulizer(model, train)  
    λ2 = λmin
    
    if length(history) == 0
        history = (
            smooth_acc=[accuracy(train, model)],
            discrete_acc=[accuracy(train, binact_model)],
            λ2=[λ2],
            ar=[ar],
            )
    end
       
    for iter in 0:periods*epochs
        i = iter % epochs + 1
        trainmode!(model) 
        for (x, y) in train
            # new train
            # gs = gradient(() -> loss(model, x, y, λ2), ps)
            gs = gradient(() -> loss(model, x, y, λ2), ps)
            Flux.update!(opt, ps, gs)
        end
        
        testmode!(model)
        acc = accuracy(train, model)
        
        binact_model = convert2binary_activation(model)
        testmode!(binact_model)
        discrete_acc = accuracy(train, binact_model)
        
        ar = activation_regulizer(model, train)

        
        λ2 = λmin + 1/2 * (λmax - λmin) * (1 + cos(i/(epochs) * π))
        
        push!(history.smooth_acc, acc)
        push!(history.discrete_acc, discrete_acc)
        push!(history.λ2, λ2)
        push!(history.ar, ar)

        # print progress
        showvalues = [
            (:acc_smooth, round(100 * history.smooth_acc[end]; digits=2)),
            (:acc_binary_activation, round(100 * history.discrete_acc[end]; digits=2)),
            (:λ2, λ2),
            (:regularization, ar),
            (:λreg, ar * λ2),
        ]
        ProgressMeter.next!(p; showvalues)
    end
    return history, λ2
end

ar = activation_regulizer(model, train_data)
ar *λmin
ar * λmax



periods = 1
epochs = 30
history = []
λmax=10e-1
λmin = 10e-6

history, λ2 = train_activation_epochs(model, AdaBelief(), train_data, objective, epochs=epochs, periods=periods,history=history, λmin=λmin, λmax=λmax)


f = show_accuracies(history)


f = Figure()
ax1 = Axis(f[1, 1])
ax2 = Axis(f[1, 1], yaxisposition = :right)
hidespines!(ax2)
hidexdecorations!(ax2)

lines!(ax1, history.smooth_acc, color = :red)
lines!(ax1, history.discrete_acc, color = :orange)
# if :ar in keys(history)
# end
lines!(ax2, history.ar, color = :darkblue)
# if :λ2 in keys(history)
#     lines!(ax2, history.λ2, color = :aqua)
# end

f
