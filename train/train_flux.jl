using Flux: params, relu, AdaBelief
using Flux: Dense, Chain

include("utils.jl")


train_data, test_data = createloader(batchsize=256)
model = Chain(
    Dense(28^2, 256, sign),
    Dense(256, 10)
)

println("accuracy before:")
accuracy(train_data, model)
accuracy(test_data, model)

train_model(model, AdaBelief(), train_data, test_data, epochs=15, loss=logitcrossentropy)

println("accuracy after:")
accuracy(train_data, model)
accuracy(test_data, model)



