using BTNNs: STLayer, get_W, STLayer, accuracy
using Flux: Chain, AdaBelief, logitcrossentropy
using BTNNs: train_st!

using CairoMakie: save, Figure, lines, lines!, hidespines!, hidexdecorations!, Axis, save, Legend, axislegend, Linestyle


using DelimitedFiles: writedlm


include("../data/datasets.jl")

train_data, test_data = createloader_MNIST()

model = Chain(
    STLayer(784, 256, tanh),
    STLayer(256, 100, tanh),
    STLayer(100, 10, tanh),
)

model_swap = Chain(
    STLayer(784, 256, tanh),
    STLayer(256, 100, tanh),
    STLayer(100, 10, tanh),
)

accuracy(train_data, model_swap)

history = train_st!(model, AdaBelief(), train_data, test_data, epochs=30)


history_swap = train_st!(model_swap, AdaBelief(), train_data, test_data, epochs=30)

history

data_swap = [history.smooth_test_acc, history_swap.smooth_test_acc]

writedlm("history/ste_bathnorm_swap.csv", data)
save("history/ste_bathnorm_swap.png", f)


f = compare_accuracy(data_swap[1, :], data_swap[2,:])

function compare_accuracy(history1, history2, show_legend=true)
    f = Figure()
    ax1 = Axis(f[1, 1], xlabel="epoch", ylabel="validation accuracy", 
    title="The difference in accuracy when chaning the order in which we apply σ and bn on MNIST.")

    lines!(ax1, history1, color = :red, label="σ ∘ bn(a)")
    # axislegend(ax1, "", position = :rb)
    lines!(ax1, history2, color = :orange, label ="bn ∘ σ(a)")

    if show_legend
        axislegend("", position = :rb)
    end
    f
end

using DelimitedFiles: readdlm

data_swap = readdlm("history/ste_bathnorm_swap.csv")
data_swap[1,:]