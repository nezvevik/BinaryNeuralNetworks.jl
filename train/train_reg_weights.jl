using BinaryNeuralNetwork: RegularizedLayer, weight_regulizer, convert2ternary_weights
using Flux: Chain, AdaBelief, mse

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
ternary_weights_model = convert2ternary_weights(model)
accuracy(train_data, ternary_weights_model)

# reality check
weight_regulizer(ternary_weights_model)

function objective(m, x, y, λ1)
    mse(m(x), y) + λ1 * weight_regulizer(m)
end

function mse_loss(m, data)
    sum(mse(m(x), y) for (x, y) in data)
end

mse(5.0, 3.0)

i = 0
for (x, y) in train_data
    println(size(model(x), 2))
    println(model(x)[1])
    println(mse(model(x), y))
    println(mse(ternary_weights_model(x), y))
    i += 1
    break
    # if i > 10
    #     break    
    # end
end

function train_activation_epochs(model, opt, train, loss; history = [],
    epochs::Int=25, periods::Int=4, λmax=10e-3, λmin=10e-6)
    
    p = Progress(periods*epochs, 1)
    ps = params(model)
    # println(model)
    ternary_weights_model = convert2ternary_weights(model)
    
    wr = weight_regulizer(model)  
    λ1 = λmin
    
    if length(history) == 0
        history = (
            smooth_acc=[accuracy(train, model)],
            discrete_acc=[accuracy(train, ternary_weights_model)],
            loss=[mse_loss(model, train)],
            λ1=[λ1],
            wr=[wr],
            )
    end
       
    for iter in 0:periods*epochs
        i = iter % epochs + 1
        trainmode!(model) 
        for (x, y) in train
            # new train
            # gs = gradient(() -> loss(model, x, y, λ1), ps)
            gs = gradient(() -> loss(model, x, y, λ1), ps)
            Flux.update!(opt, ps, gs)
        end
        
        testmode!(model)
        acc = accuracy(train, model)
        
        ternary_weights_model = convert2ternary_weights(model)
        testmode!(ternary_weights_model)
        discrete_acc = accuracy(train, ternary_weights_model)
        
        wr = weight_regulizer(model)

        
        λ1 = λmin + 1/2 * (λmax - λmin) * (1 + cos(i/(epochs) * π))
        
        push!(history.smooth_acc, acc)
        push!(history.discrete_acc, discrete_acc)
        push!(history.λ1, λ1)
        push!(history.wr, wr)

        # print progress
        showvalues = [
            (:acc_smooth, round(100 * history.smooth_acc[end]; digits=2)),
            (:acc_binary_activation, round(100 * history.discrete_acc[end]; digits=2)),
            (:loss, mse_loss(model, train)),
            (:λ1, λ1),
            (:regularization, wr),
            (:λreg, wr * λ1),
        ]
        ProgressMeter.next!(p; showvalues)
    end
    return history, λ1
end


wr = weight_regulizer(model)
p = round(log10(wr))

periods = 1
epochs = 20
history = []
λmin = 10^(- p - 2)
# λmax = λmin * 10^3
λmax = 10e-3

wr * λmin
wr * λmax

λmin = 0.
λmax = 0.

history, λ1 = train_activation_epochs(model, AdaBelief(), train_data, objective, epochs=epochs, periods=periods,history=history, λmin=λmin, λmax=λmax)



f = show_accuracies(history)
f = show_regularization(history)
