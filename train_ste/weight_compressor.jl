using BTNNs: STLayer, get_W, STLayer
using Flux: Chain, AdaBelief
using Flux: gradient, params, logitcrossentropy
using BTNNs: train_st!

using CairoMakie: save, Figure, lines, lines!, hidespines!, hidexdecorations!, Axis, save, Legend, axislegend, Linestyle


using DelimitedFiles: writedlm


include("../data/datasets.jl")

train_data, test_data = createloader_MNIST()

model_default = Chain(
    STLayer(784, 256, tanh, identity),
    STLayer(256, 100, tanh, identity),
    STLayer(100, 10, tanh, identity),
)
model_activation = Chain(
    STLayer(784, 256, tanh, identity),
    STLayer(256, 100, tanh, identity),
    STLayer(100, 10, tanh, identity),
)

model= Chain(
    STLayer(784, 256, tanh, tanh),
    STLayer(256, 100, tanh, tanh),
    STLayer(100, 10, tanh, tanh),
)


default_history = train_st!(model_default, AdaBelief(), train_data, test_data, epochs=30)


activation_history = train_st!(model_activation, AdaBelief(), train_data, test_data, epochs=30)


history = train_st!(model, AdaBelief(), train_data, test_data, epochs=30)


data = [default_history.smooth_test_acc, activation_history.smooth_test_acc, history.smooth_test_acc]

writedlm("history/ste_nonlinearity.csv", data)

f = show_all_accuracies(data)

function show_all_accuracies(history, show_legend=true)
    f = Figure()
    ax1 = Axis(f[1, 1], xlabel="epoch", ylabel="validation accuracy", title="Accuracies of the continuous and the converted (discrete) model on the MNIST dataset")

    lines!(ax1, history[1], color = :red, label="continuous model")
    # axislegend(ax1, "", position = :rb)
    lines!(ax1, history[2], color = :orange, label ="converted (discrete) model")
    lines!(ax1, history[3], color = :blue, label ="converted (discrete) model")

    if show_legend
        axislegend("", position = :rb)
    end
    f
end