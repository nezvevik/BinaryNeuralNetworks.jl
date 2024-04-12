using BinaryNeuralNetwork: RegularizedLayer, weight_regulizer
using BinaryNeuralNetwork: convert2binary_activation, convert2ternary_weights, convert2discrete
using Flux: Chain, crossentropy

model = Chain(
    RegularizedLayer(28^2, 256, tanh, tanh),
    RegularizedLayer(256, 256, tanh, tanh),
    RegularizedLayer(256, 10, tanh, tanh)
)


include("utils.jl")
train_data, test_data = createloader(batchsize=256)

discrete_model = convert2ternary_weights(model)
accuracy(train_data, discrete_model)
weight_regulizer(discrete_model)

# get data
function sum_loss(m, data)
    sum(logitcrossentropy(m(x), y) for (x, y) in data)
end


wr = weight_regulizer(model)
p1 = round(log10(wr))

l = sum_loss(model, train_data)

# initialization of hyperparameters
history = []
periods = 1
epochs = 10

λ1min = 10^(-p1)
λ1max = l / wr

# check
λ1min * wr
λ1max * wr


function objective(m, x, y, λ1)
    # Flux.Zygote.@ignore push!(history.ar, ar)
    o, r = m((x, 0f0))
    logitcrossentropy(o, y) + λ1 * weight_regulizer(m)
end

history = train_reg_weights(
    model, AdaBelief(), train_data, objective;
    history=history, periods=periods, epochs=epochs,
    λ1min=λ1min, λ1max=λ1max)

show_accuracies(history)
show_regularization(history)

function train_reg_weights(model, opt, train, loss; history = [],
    periods::Int=4, epochs::Int=25,
    λ1min=10e-6, λ1max=10e-3)
    
    p = Progress(periods*epochs, 1)
    ps = params(model)

    tern_w_model = convert2ternary_weights(model)
    testmode!(tern_w_model)
    
    wr = weight_regulizer(model)  
    λ1 = λ1min
    
    if length(history) == 0
        history = (
            smooth_acc=[accuracy(train, model)],
            discrete_acc=[accuracy(train, tern_w_model)],
            loss=[sum_loss(model, train)],
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
        
        tern_w_model = convert2ternary_weights(model)
        testmode!(tern_w_model)
        discrete_acc = accuracy(train, tern_w_model)
        

        wr = weight_regulizer(model)

        
        λ1 = λ1min + 1/2 * (λ1max - λ1min) * (1 + cos(i/(epochs) * π))
        
        push!(history.smooth_acc, acc)
        push!(history.discrete_acc, discrete_acc)
        push!(history.loss, sum_loss(model, train))
        push!(history.λ1, λ1)
        push!(history.wr, wr)

        # print progress
        showvalues = [
            (:acc_smooth, round(100 * history.smooth_acc[end]; digits=2)),
            (:acc_binary_activation, round(100 * history.discrete_acc[end]; digits=2)),
            (:loss, round(history.loss[end]; digits=4)),
            (:λ1, λ1),
            (:regularization, wr),
            (:λreg, wr * λ1),
        ]
        ProgressMeter.next!(p; showvalues)
    end
    return history
end


