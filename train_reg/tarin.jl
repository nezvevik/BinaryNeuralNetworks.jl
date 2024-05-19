using BinaryNeuralNetwork: RegularizedLayer, weight_regulizer, activation_regulizer, get_binary_regulizer, get_ternary_regulizer
using BinaryNeuralNetwork: convert2binary_activation, convert2ternary_weights, convert2discrete, set_β, set_τ
using Flux: Chain, sum, logitcrossentropy, mse, mean


β = get_binary_regulizer(2)
τ = get_ternary_regulizer(3)

model = Chain(
    RegularizedLayer(28^2, 256, tanh, tanh, β, τ),
    RegularizedLayer(256, 10, tanh, tanh, β, τ)
)

model = set_β(model, β)
model = set_τ(model, τ)

include("utils.jl")
train_data, test_data = createloader(batchsize=256)

loss = logitcrossentropy
loss = mse

function mean_loss(m, data)
    mean(loss(m(x), y) for (x, y) in data)
end


function objective(m, x, y, λ1, λ2)
    o, r = m((x, 0f0))
    loss(o, y) + λ1 * weight_regulizer(m) + λ2 * r
    # λ2 * r
end



history = []
periods = 1
epochs = 15

l = mean_loss(model, train_data)

# set λs 
wr = weight_regulizer(model)
ar = activation_regulizer(model, train_data)

λ1min, λ1max = get_λ_range(wr, l)
λ2min, λ2max = get_λ_range(ar, l)

# _, λ2min = get_λ_range(ar, l)

(λ1min, λ1max) .* wr
(λ2min, λ2max) .* ar

update_λ1(λ1min, λ1max, i, epochs) = 0f0
update_λ1(λ1min, λ1max, i, epochs) = λ1min + 1/2 * (λ1max - λ1min) * (1 + cos(i/(epochs) * π))
update_λ2(λ2min, λ2max, i, epochs) = 0f0
update_λ2(λ2min, λ2max, i, epochs) = λ2min + 1/2 * (λ2max - λ2min) * (1 + cos(i/(epochs) * π))


history = train_reg!(model, AdaBelief(), train_data, objective;
    history=history, periods=periods, epochs=epochs,
    update_λ1=update_λ1,
    update_λ2=update_λ2,
    λ1min=λ1min, λ1max=λ1max,
    λ2min=λ2min, λ2max=λ2max)


# plot the resluts
f = show_all_accuracies(history)
f = show_regularization(history)
f = show_ar(history)
f = show_wr(history)


using CairoMakie: save

save("wr.png", f)