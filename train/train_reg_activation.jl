using BinaryNeuralNetwork: RegularizedLayer, weight_regulizer, activation_regulizer
using BinaryNeuralNetwork: convert2binary_activation, convert2ternary_weights, convert2discrete
using Flux: Chain, mse, crossentropy

model = Chain(
    RegularizedLayer(28^2, 256, tanh, identity),
    RegularizedLayer(256, 256, tanh, identity),
    RegularizedLayer(256, 10, tanh, identity)
)


include("utils.jl")
train_data, test_data = createloader(batchsize=256)

binact_model = convert2binary_activation(model)
accuracy(train_data, binact_model)
activation_regulizer(binact_model, train_data)

# get data
function mse_loss(m, data)
    sum(logitcrossentropy(m(x), y) for (x, y) in data)
end


ar = activation_regulizer(model, train_data)
p2 = round(log10(ar))

l = mse_loss(model, train_data)

# initialization of hyperparameters
history = []
periods = 1
epochs = 10

λ2min = 10^(- p2)
λ2max = l / ar * 4

# check
λ2min * ar
λ2max * ar


function objective(m, x, y, λ2)
    # Flux.Zygote.@ignore push!(history.ar, ar)
    o, r = m((x, 0f0))
    logitcrossentropy(o, y) + λ2 * r
end

history = train_reg_activation(
    model, AdaBelief(), train_data, objective;
    history=history, periods=periods, epochs=epochs,
    λ2min=λ2min, λ2max=λ2max)

show_accuracies(history)
show_regularization(history)

function train_reg_activation(model, opt, train, loss; history = [],
    periods::Int=4, epochs::Int=25,
    λ2min=10e-6, λ2max=10e-3)
    
    p = Progress(periods*epochs, 1)
    ps = params(model)

    binact_model = convert2binary_activation(model)
    testmode!(binact_model)
    
    ar = activation_regulizer(model, train)  
    λ2 = λ2min
    
    if length(history) == 0
        history = (
            smooth_acc=[accuracy(train, model)],
            discrete_acc=[accuracy(train, binact_model)],
            loss=[mse_loss(model, train)],
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

        
        λ2 = λ2min + 1/2 * (λ2max - λ2min) * (1 + cos(i/(epochs) * π))
        
        push!(history.smooth_acc, acc)
        push!(history.discrete_acc, discrete_acc)
        push!(history.loss, mse_loss(model, train))
        push!(history.λ2, λ2)
        push!(history.ar, ar)

        # print progress
        showvalues = [
            (:acc_smooth, round(100 * history.smooth_acc[end]; digits=2)),
            (:acc_binary_activation, round(100 * history.discrete_acc[end]; digits=2)),
            (:loss, round(history.loss[end]; digits=4)),
            (:λ2, λ2),
            (:regularization, ar),
            (:λreg, ar * λ2),
        ]
        ProgressMeter.next!(p; showvalues)
    end
    return history
end


