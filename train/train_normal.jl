using BinaryNeuralNetwork: Layer
using Flux: AdaBelief, relu, Chain

include("utils.jl")

train_data, test_data = createloader(batchsize=256)

model = Chain(
    Layer(28^2, 256, sign),
    Layer(256, 10, identity)
)


accuracy(train_data, model)
accuracy(test_data, model)

history = train_model(model, AdaBelief(), train_data, test_data; epochs=15)

println("accuracy after training")
accuracy(train_data, model)
accuracy(test_data, model)



