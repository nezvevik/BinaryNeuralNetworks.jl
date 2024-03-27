using BinaryNeuralNetwork: RegularizedLayer, activation_regulizer
using BinaryNeuralNetwork: convert2binary_weights, convert2binary_activation
using Flux: Chain, AdaBelief, mse

# Makie


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

binact_model = convert2binary_activation(model)
accuracy(train_data, binact_model)

ar = activation_regulizer(model, train_data)


λ2 = 10^(-6)

function objective(m, x, y, λ2::Float64=0.00000001)
    logitcrossentropy(m(x), y) + λ2 * activation_regulizer(m, x)
end

# train_model_activation(model, AdaBelief(), train_data, objective, epochs=30, λ2 = λ2, should_increase=true)
history, λ2 = train_model_activation(model, AdaBelief(), train_data, objective, epochs=25, λ2 = λ2, should_increase=true)


plot([history.smooth_acc, history.binact_acc])
plt = plot!(twinx(),history.λ2,color=:red,xticks=:none,label="λ2")

plot(plt,history.λreg, color=:green, label="activation regulizer")

plt

savefig(plt, "constant facotr 85-95.png")
