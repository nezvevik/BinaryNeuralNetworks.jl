using BinaryNeuralNetwork: RegularizedLayer, weight_regulizer, activation_regulizer
using BinaryNeuralNetwork: convert2ternary_weights, convert2binary_activation, convert2discrete

using Flux: Chain, AdaBelief

include("utils.jl")

# get data
train_data, test_data = createloader(batchsize=256)

# init model
# insert identity as weight regulizer
model = Chain(
    RegularizedLayer(28^2, 256, tanh, tanh),
    RegularizedLayer(256, 256, tanh, tanh),
    RegularizedLayer(256, 10, tanh, tanh)
)


# default accuracy ≈ 0.1
accuracy(train_data, model)
accuracy(test_data, model)

# check default binary model accuracy
discrete_model = convert2discrete(model)
accuracy(train_data, discrete_model)


function objective(m, x, y, λ1::Float64=0.00000001, λ2::Float64=0.00000001)
    mse(m(x), y) + λ1 * weight_regulizer(m) + λ2 * activation_regulizer(m, x)
end

# TODO early stopping

function train_regularization(model, opt, train, loss; history = [],
    epochs::Int=25, periods::Int=4,
    λ1max=10e-3, λ1min=10e-6,
    λ2max=10e-3, λ2min=10e-6)
    
    p = Progress(periods*epochs, 1)
    ps = params(model)
    # println(model)
    discrete_model = convert2discrete(model)
    
    wr = weight_regulizer(model)  
    ar = activation_regulizer(model, train)  
    λ1 = λ1min
    λ2 = λ2min
    
    if length(history) == 0
        history = (
            smooth_acc=[accuracy(train, model)],
            discrete_acc=[accuracy(train, discrete_model)],
            λ1=[λ1],
            λ2=[λ2],
            wr=[wr],
            ar=[ar],
            )
    end
       
    for iter in 0:periods*epochs
        i = iter % epochs + 1
        trainmode!(model) 
        for (x, y) in train
            # new train
            # gs = gradient(() -> loss(model, x, y, λ1), ps)
            gs = gradient(() -> loss(model, x, y, λ1, λ2), ps)
            Flux.update!(opt, ps, gs)
        end
        
        testmode!(model)
        acc = accuracy(train, model)
        
        discrete_model = convert2discrete(model)
        testmode!(discrete_model)
        discrete_acc = accuracy(train, discrete_model)
        
        wr = weight_regulizer(model)
        ar = activation_regulizer(model, train)

        λ1 = λ1min + 1/2 * (λ1max - λ1min) * (1 + cos(i/(epochs) * π))
        λ2 = λ2min + 1/2 * (λ2max - λ2min) * (1 + cos(i/(epochs) * π))
        
        push!(history.smooth_acc, acc)
        push!(history.discrete_acc, discrete_acc)
        push!(history.λ1, λ1)
        push!(history.λ2, λ2)
        push!(history.wr, wr)
        push!(history.ar, ar)

        # print progress
        showvalues = [
            (:acc_smooth, round(100 * history.smooth_acc[end]; digits=2)),
            (:discrete_acc, round(100 * history.discrete_acc[end]; digits=2)),
            (:λ1, λ1),
            (:λ2, λ2),
            (:wr, wr),
            (:ar, ar),
            (:λ1wr, λ1 * wr),
            (:λ2ar, λ2 * ar),
        ]
        ProgressMeter.next!(p; showvalues)
    end
    return history
end


wr = weight_regulizer(model)
ar = activation_regulizer(model, train_data)

p1 = round(log10(wr))
p2 = round(log10(ar))

λ1min = 10^(- p1 - 1)
λ1max = λ1min * 10^3

λ2min = 10^(- p2 - 1)
λ2max = λ2min * 10^3


λ1min*wr
λ1max*wr

λ2min*ar
λ2max*ar

λ1min = 0.0
λ1max = 10e-3
λ2min = 0.
λ2max = 0.

periods = 1
epochs = 10
history = []

history =  train_regularization(model, AdaBelief(), train_data, objective, epochs=epochs, periods=periods, history=history, λ1min=λ1min, λ1max=λ1max, λ2min=λ2min, λ2max=λ2max)



f = show_accuracies(history)
f = show_regularization(history)
