using BinaryNeuralNetwork: PSALayer, psa_binary, psa_ternary, convert2discrete, STELayer
using Flux: Chain, relu, logitcrossentropy
using Random


include("utils.jl")

train_data, test_data = createloader(batchsize=256)

Random.seed!(37)
model = Chain(
    PSALayer(28^2, 256, identity, true),
    PSALayer(256, 10, identity, true)
)

model = Chain(
    STELayer(28^2, 256, tanh, true),
    STELayer(256, 10, tanh, true)
)

accuracy(train_data, model)
accuracy(test_data, model)

train_psa!(model, AdaBelief(), train_data, test_data, epochs=15, loss=logitcrossentropy)

ps = params(model)
x , y = first(train_data)
gs = gradient(() -> logitcrossentropy(model(x), y), ps)


gradient(() -> logitcrossentropy(model(x), y), ps)

function train_psa!(model, opt, train, test; loss=logitcrossentropy, epochs::Int=30)
    p = Progress(epochs, 1)
    ps = params(model)
    history = (
        train_acc=[accuracy(train, model)],
        test_acc=[accuracy(test, model)],
    )

    for _ in 1:epochs
        for (x, y) in train
            # new train
            gs = gradient(() -> loss(model(x), y), ps)
            Flux.update!(opt, ps, gs)
        end

        # compute accuracy  
        push!(history.train_acc, accuracy(train, model))
        push!(history.test_acc, accuracy(test, model))

        # print progress
        showvalues = [
            (:acc_train_0, round(100 * history.train_acc[1]; digits=2)),
            (:acc_train, round(100 * history.train_acc[end]; digits=2)),
            (:acc_test_0, round(100 * history.test_acc[1]; digits=2)),
            (:acc_test, round(100 * history.test_acc[end]; digits=2)),
        ]
        ProgressMeter.next!(p; showvalues)
    end
    return history
end


discrete = convert2discrete(model)

accuracy(train_data, discrete)
accuracy(test_data, discrete)
