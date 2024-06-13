using BTNNs: RLayer, convert2discrete, DSTLayer
using BTNNs: train_default!, error, accuracy, get_W
using CairoMakie: save, Figure, lines, lines!, hidespines!, hidexdecorations!, Axis, save, Legend, axislegend, Linestyle

using Flux: Chain, hardtanh, AdaBelief
using DelimitedFiles: writedlm

include("../data/datasets.jl")

model = Chain(
    DSTLayer(784, 256, tanh),
    DSTLayer(256, 100, tanh),
    DSTLayer(100, 10, tanh),
)

train_data, test_data = createloader_MNIST()
accuracy(train_data, model)
accuracy(test_data, model)

history = train_default!(model,AdaBelief(), train_data, test_data; epochs=5)


discrete = convert2discrete(model)

get_W(discrete[1])
x = first(train_data)[1]
discrete(x)
model(x)
accuracy(test_data, discrete)'

writedlm("zdefault_conversion.csv", history)

f = show_all_accuracies(history)
save("history/default_conversion.png", f)

function show_all_accuracies(history, show_legend=true)
    f = Figure()
    ax1 = Axis(f[1, 1], xlabel="epoch", ylabel="validation accuracy", title="Accuracies of the continuous and the converted (discrete) model on the MNIST dataset")

    lines!(ax1, history.smooth_test_acc, color = :red, label="continuous model")
    # axislegend(ax1, "", position = :rb)
    lines!(ax1, history.discrete_test_acc, color = :orange, label ="converted (discrete) model")

    if show_legend
        axislegend("", position = :rb)
    end
    f
end