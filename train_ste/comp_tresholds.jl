using BTNNs: STLayer, get_W, STLayer, accuracy
using Flux: Chain, AdaBelief, logitcrossentropy
using BTNNs: train_st!

using CairoMakie: save, Figure, lines, lines!, hidespines!, hidexdecorations!, Axis, save, Legend, axislegend, Linestyle


using DelimitedFiles: writedlm


include("../data/datasets.jl")

train_data, test_data = createloader_MNIST()

default_model = Chain(
    STLayer(784, 256, tanh),
    STLayer(256, 100, tanh),
    STLayer(100, 10, tanh),
)

model_6 = Chain(
    STLayer(784, 256, tanh),
    STLayer(256, 100, tanh),
    STLayer(100, 10, tanh),
)

accuracy(train_data, default_model)

history = train_st!(default_model, AdaBelief(), train_data, test_data, epochs=30)


history_6 = train_st!(model_6, AdaBelief(), train_data, test_data, epochs=30)

history

data_comp = [history.smooth_test_acc, history_swap.smooth_test_acc]

writedlm("history/ste_bathnorm_swap.csv", data)
save("history/ste_bathnorm_swap.png", f)


f = compare_accuracy(history, history_6)

function compare_accuracy(history1, history2, show_legend=true)
    f = Figure()
    ax1 = Axis(f[1, 1], xlabel="epoch", ylabel="validation accuracy", 
    title="Difference in accuracy when chaning the order in which we apply σ and batchnormalization.")

    lines!(ax1, history1.smooth_test_acc, color = :red, label="appying activation after batchnormalization, σ ∘ bn(x)")
    # axislegend(ax1, "", position = :rb)
    lines!(ax1, history2.smooth_test_acc, color = :orange, label ="appying activation before batchnormalization, bn ∘ σ(x)")

    if show_legend
        axislegend("", position = :rb)
    end
    f
end