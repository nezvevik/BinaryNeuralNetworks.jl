using BinaryNeuralNetwork: RegularizedLayer, weight_regulizer, activation_regulizer, set_β, binary_regulizer
using BinaryNeuralNetwork: convert2binary_activation, convert2ternary_weights, convert2discrete
using Flux: Chain, sum, logitcrossentropy, Zygote, AdaBelief


include("utils.jl")
train_data, test_data = createloader(batchsize=256)

default_model = Chain(
    RegularizedLayer(28^2, 256, tanh, tanh),
    RegularizedLayer(256, 10, tanh, tanh)
)

function objective(m, x, y, λ1, λ2, ar_list=[])
    o, r = m((x, 0f0))
    if ar_list != []
        Zygote.@ignore push!(ar_list, r)
    end
    loss(o, y) + λ2 * r
end

loss = logitcrossentropy

function sum_loss(m, data)
    sum(loss(m(x), y) for (x, y) in data)
end


βs = [("abs", 1.0), ("pow", 1.0), ("pow", 2.0), ("pow", 4.0), ("pow", 10.0)]
histories = []
for (t, n) in βs
    β = get_binary_regulizer(t, n)

    model = set_β(deepcopy(default_model), β)
    println("next model β: $t, $n")

    # train the model
    history = []
    periods = 3
    epochs = 20

    ar = activation_regulizer(model, train_data)
    l = sum_loss(model, train_data)

    λ1min, λ1max = (0f0,0f0)
    λ2min, λ2max = get_λ_range(ar, l) .* 30


    update_λ1(λ1min, λ1max, i, epochs) = 0f0
    update_λ2(λ2min, λ2max, i, epochs) = λ2min + 1/2 * (λ2max - λ2min) * (1 + cos(i/(epochs) * π))

    history = train_reg_activation!(model, AdaBelief(), train_data, objective;
        history=history, periods=periods, epochs=epochs,
        update_λ1=update_λ1,
        update_λ2=update_λ2,
        λ1min=λ1min, λ1max=λ1max,
        λ2min=λ2min, λ2max=λ2max)

    push!(histories, history)

end

histories
show_all_accuracies(histories[1])
show_ar(history)
history.smooth_acc

periods = 2
epochs = 3
history = []

history = train_reg_activation!(model, AdaBelief(), train_data, objective;
        history=history, periods=periods, epochs=epochs,
        update_λ1=update_λ1,
        update_λ2=update_λ2,
        λ1min=λ1min, λ1max=λ1max,
        λ2min=λ2min, λ2max=λ2max)







β = binary_regulizer("pow", 1.0)
model = Chain(
    RegularizedLayer(28^2, 256, tanh, tanh, β),
    RegularizedLayer(256, 10, tanh, tanh, β)
)



model = set_β(model, β)
loss = logitcrossentropy

function sum_loss(m, data)
    sum(loss(m(x), y) for (x, y) in data)
end


function objective(m, x, y, λ1, λ2, ar_list=[])
    o, r = m((x, 0f0))
    if ar_list != []
        Flux.Zygote.@ignore push!(ar_list, r)
    end
    loss(o, y) + λ1 * weight_regulizer(m) + λ2 * r
end



history = []
periods = 2
epochs = 30

l = sum_loss(model, train_data)

# set λs 
wr = weight_regulizer(model)
ar = activation_regulizer(model, train_data)

λ1min, λ1max = get_λ_range(wr, l) .* (1 / 1000)
λ2min, λ2max = get_λ_range(ar, l) .* 4

(λ1min, λ1max) .* wr
(λ2min, λ2max) .* ar

update_λ1(λ1min, λ1max, i, epochs) = 0f0
update_λ2(λ2min, λ2max, i, epochs) = 0f0
update_λ2(λ2min, λ2max, i, epochs) = λ2min + 1/2 * (λ2max - λ2min) * (1 + cos(i/(epochs) * π))


history = train_reg!(model, AdaBelief(), train_data, objective;
    history=history, periods=periods, epochs=epochs,
    update_λ1=update_λ1,
    update_λ2=update_λ2,
    λ1min=λ1min, λ1max=λ1max,
    λ2min=λ2min, λ2max=λ2max)

# plot the resluts
show_all_accuracies(history, false)
show_regularization(history)


