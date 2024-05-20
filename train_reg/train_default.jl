using BinaryNeuralNetwork: BTLayer
using BinaryNeuralNetwork: weight_regularizer, activation_regularizer, get_binary_regularizer, set_output_regularizer
using Flux: Chain

include("../data/datasets.jl")


l = BTLayer(10, 10)

reg = get_binary_regularizer(1.0)

model = Chain(
    BTLayer(28^2, 256, tanh),
    BTLayer(256, 100, tanh),
    BTLayer(100, 10, tanh)
)

train_data, test_data = createloader_MNIST()


set_output_regularizer(model, reg)
model[1].output_regularizer


@time activation_regularizer(model, train_data)
@time activation_regularizer(model, test_data)

