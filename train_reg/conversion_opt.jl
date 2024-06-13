using BTNNs: RLayer, accuracy, ternary_quantizer, get_ternary_quantizer, weight_regularizer, activation_regularizer
using BTNNs: convert2discrete, convert2ternary_weights, convert2binary_activation
using BTNNs: set_weight_quantizer, train_reg_no_log!
using Flux: Chain, AdaBelief, mean, mse, logitcrossentropy, sum, params, gradient, update!, trainmode!, testmode!
using ProgressMeter

using CairoMakie: hidedecorations!,scatter!, save, Figure, lines, lines!, hidespines!, hidexdecorations!, Axis, save, Legend, axislegend, Linestyle

include("../data/datasets.jl")

train_data, test_data = createloader_MNIST()

model = Chain(
    RLayer(28^2, 256, tanh),
    RLayer(256, 10, tanh)
)

loss=logitcrossentropy

function mean_loss(m, data)
    mean(loss(m(x), y) for (x, y) in data)
end

function objective(m, x, y, λ1, λ2)
    o, r = m((x, 0f0))
    loss(o, y) + λ1 * weight_regularizer(m) + λ2 * r
    # loss(o, y)
end

function get_λ_range(reg, l)
    pow = round(log10(reg))
    # return 10^(- pow), l / reg
    return 0f0, l / reg
end

l = mean_loss(model, train_data)

# set λs 
wr = weight_regularizer(model)
ar = activation_regularizer(model, train_data)

λ1min, λ1max = get_λ_range(wr, l) .* 10 
λ2min, λ2max = get_λ_range(ar, l) .* 10

history = []
periods = 1
epochs = 8

update_λ1(λ1min, λ1max, i, epochs) = 0f0
update_λ1(λ1min, λ1max, i, epochs) = λ1min + 1/2 * (λ1max - λ1min) * (1 + cos(i/(epochs) * π))
update_λ2(λ2min, λ2max, i, epochs) = 0f0
update_λ2(λ2min, λ2max, i, epochs) = λ2min + 1/2 * (λ2max - λ2min) * (1 + cos(i/(epochs) * π))


history = train_reg_no_log!(model, AdaBelief(), train_data, objective;
    history=history, periods=periods, epochs=epochs,
    update_λ1=update_λ1,
    update_λ2=update_λ2,
    λ1min=λ1min, λ1max=λ1max,
    λ2min=λ2min, λ2max=λ2max)




history


T = (5:30)



for i in T2
    println(i/40)
end

T2 = [0.2, 0.25, 0.28, 0.3,0.32, 0.35, 0.4, 0.45, 0.5]
T1 = T2 .* (-1)

train_acc = zeros(Float32, length(T1), length(T2))
test_acc = zeros(Float32, length(T1), length(T2))

best_t1, best_t2, best_acc = 0, 0, 0

train_acc
best_t1, best_t2, best_acc

for (i, t1) in enumerate(T1)
    for (j, t2) in enumerate(T2)
        if t1 >= t2 continue end
        set_weight_quantizer(model, get_ternary_quantizer(t1, t2))
        discrete = convert2discrete(model)
        acc = accuracy(train_data, discrete)
        test_a = accuracy(test_data, discrete)

        train_acc[i, j] = acc
        test_acc[i, j] = test_a

        if acc > best_acc
            best_acc = acc
            best_t1 = t1
            best_t2 = t2
        end
    end
end

using Flux: diag


round.(100 .* train_acc; digits=2)
round.(100 .* test_acc; digits=2)

train_acc_tmp = train_acc 
test_acc_tmp = test_acc

best_t1, best_t2

diag(train_acc)   
diag(test_acc) 



T = (5:30)
accuracies = zeros(Float32, 2, length(T))

best_i, best_t, best_acc = 1, 0, 0


accuracies

for (i, t) in enumerate(T)
    treshold = t / 40
    set_weight_quantizer(model, get_ternary_quantizer(-treshold, treshold))
    discrete = convert2discrete(model)
    acc = accuracy(train_data, discrete)
    test_a = accuracy(test_data, discrete)

    accuracies[1, i] = acc
    accuracies[2, i] = test_a

    if acc > best_acc
        best_acc = acc
        best_t = treshold
        best_i = i
    end
end
accuracies

best_t
best_acc
best_i
accuracies[1, best_i]
accuracies[2, best_i]#


set_weight_quantizer(model, get_ternary_quantizer(-best_t, best_t))
discrete = convert2discrete(model)
acc = accuracy(train_data, discrete)
test_a = accuracy(test_data, discrete)


T
f = scatter_comp(T, accuracies)
save("scatter_comp.png", f)
writedlm("history/history_reg.csv", history)


function scatter_comp(T, accuracies)
    f= Figure()
    ax1 = Axis(f[1, 1], xlabel="treshold value", ylabel="accuracy",
    title="The impact of tresholds on the discrete accuracy on MNIST dataset.")
    scatter!(ax1, T, accuracies[1,:], color=:red, label="train")
    scatter!(ax1, T, accuracies[2,:], color=:orange, label="test")

    ax2 = Axis(f[1, 1])
    lines!(ax2, accuracies[1,:], color=:red, label="discrete train accuracy")
    lines!(ax2, accuracies[2,:], color=:orange, label="discrete test accuracy")
    hidedecorations!(ax2)

    axislegend("", position = :lb)
    f

end
