using Flux: gradient, train!, params, onehotbatch, flatten, onecold, logitcrossentropy, mse, update!
using Flux
using MLDatasets, CUDA
using Flux.Data: DataLoader
using ProgressMeter
using CairoMakie: Figure, lines, lines!, hidespines!, hidexdecorations!, Axis, save, Legend, axislegend, Linestyle
using Statistics: mean

function classify(model, x)
    ŷ = model(x)
    idxs = Tuple.(argmax(ŷ, dims=1))
    return vec(Float32.(first.(idxs)) .- 1.0)
end

function accuracy(data_loader, model)
    acc = 0
    num = 0
    for (x, y) in data_loader
        acc += sum(onecold(model(x)) .== onecold(y))
        num += size(x)[end]
    end
    return acc / num
end

function createloader(dataset=MLDatasets.MNIST; batchsize::Int=256)
    xtrain, ytrain = dataset(:train)[:]
    train_loader = DataLoader(
        (flatten(xtrain), onehotbatch(ytrain, 0:9));
        batchsize,
        shuffle=true,
    )

    xtest, ytest = dataset(:test)[:]
    test_loader = DataLoader(
        (flatten(xtest), onehotbatch(ytest, 0:9));
        batchsize,
    )
    return train_loader, test_loader
end


update_λ1(λ1min, λ1max, i, epochs) = λ1min + 1/2 * (λ1max - λ1min) * (1 + cos(i/(epochs) * π))
update_λ2(λ2min, λ2max, i, epochs) = λ2min + 1/2 * (λ2max - λ2min) * (1 + cos(i/(epochs) * π))

mean_loss(m, data) = sum(loss(m(x), y) for (x, y) in data)

function get_λ_range(reg, l)
    pow = round(log10(reg))
    # return 10^(- pow), l / reg
    return 0f0, l / reg
end

function train_reg_activation!(model, opt, train, objective;
    loss=logitcrossentropy,
    history = [],
    update_λ1=update_λ1, update_λ2=update_λ2,
    weight_regulizer=weight_regulizer, activation_regulizer=activation_regulizer,
    periods::Int=4, epochs::Int=25,
    λ1min=10e-6, λ1max=10e-3,
    λ2min=10e-6, λ2max=10e-3)
    
    p = Progress(periods*epochs, 1)
    ps = params(model)

    

    # create discrete models
    discrete_model = convert2discrete(model)
    testmode!(discrete_model)

    tern_w_model = convert2ternary_weights(model)
    testmode!(tern_w_model)
    
    bin_act_model = convert2binary_activation(model)
    testmode!(bin_act_model)
    
    # wr = weight_regulizer(model)  
    wr = 0f0
    ar = activation_regulizer(model, train)

    λ1 = update_λ1(λ1min, λ1max, 0, epochs)
    λ2 = update_λ2(λ2min, λ2max, 0, epochs)
    
    if length(history) == 0
        history = (
            smooth_acc=[accuracy(train, model)],
            discrete_acc=[accuracy(train, discrete_model)],
            tern_w_acc=[accuracy(train, tern_w_model)],
            bin_act_acc=[accuracy(train, bin_act_model)],
            loss=[mean_loss(model, train)],
            λ1=[λ1],
            wr=[wr],
            λ2=[λ2],
            ar=[ar],
            )
    end
       

    for period in 1:periods
        for e in 1:epochs
            ar_list = [0f0]
            trainmode!(model)
            for (x, y) in train
                # TODO new train
                gs = gradient(() -> objective(model, x, y, λ1, λ2, ar_list), ps)
                Flux.update!(opt, ps, gs)
            end
            
            testmode!(model)
            acc = accuracy(train, model)
            
            # get accuracies
            # discrete
            discrete_model = convert2discrete(model)
            testmode!(discrete_model)
            discrete_acc = accuracy(train, discrete_model)
            
            # weights
            tern_w_model = convert2ternary_weights(model)
            testmode!(tern_w_model)
            tern_w_acc = accuracy(train, tern_w_model)
            
            # get accuracies
            bin_act_model = convert2binary_activation(model)
            testmode!(bin_act_model)
            bin_act_acc = accuracy(train, bin_act_model)
            
            # get regularizations
            # wr = weight_regulizer(model)
            wr = 0f0
            
            ar = sum(ar_list) / (length(ar_list) - 1)
            
            # update λ
            λ1 = update_λ1(λ1min, λ1max, e, epochs)
            λ2 = update_λ2(λ2min, λ2max, e, epochs)
            
            push!(history.smooth_acc, acc)
            push!(history.discrete_acc, discrete_acc)
            push!(history.tern_w_acc, tern_w_acc)
            push!(history.bin_act_acc, bin_act_acc)
            push!(history.loss, mean_loss(model, train))
            push!(history.λ1, λ1)
            push!(history.wr, wr)
            push!(history.λ2, λ2)
            push!(history.ar, ar)
            
            # print progress
            showvalues = [
                (:period, period),
                (:epoch, e),
                (:smooth_acc, round(100 * history.smooth_acc[end]; digits=2)),
                (:discrete_acc, round(100 * history.discrete_acc[end]; digits=2)),
                (:ternary_weights_acc, round(100 * history.tern_w_acc[end]; digits=2)),
                (:binary_activation_acc, round(100 * history.bin_act_acc[end]; digits=2)),
                (:loss, round(history.loss[end]; digits=4)),
                (:λ1, λ1),
                (:wr, wr),
                (:λ1reg, wr * λ1),
                (:λ2, λ2),
                (:ar, ar),
                (:λ2reg, ar * λ2),
                ]
                ProgressMeter.next!(p; showvalues)
            end
        end
    return history
end




function train_reg!(model, opt, train, objective;
    loss=logitcrossentropy,
    history = [],
    update_λ1=update_λ1, update_λ2=update_λ2,
    weight_regulizer=weight_regulizer, activation_regulizer=activation_regulizer,
    periods::Int=4, epochs::Int=25,
    λ1min=10e-6, λ1max=10e-3,
    λ2min=10e-6, λ2max=10e-3)
    
    p = Progress(periods*epochs, 1)
    ps = params(model)

    

    # create discrete models
    discrete_model = convert2discrete(model)
    testmode!(discrete_model)

    tern_w_model = convert2ternary_weights(model)
    testmode!(tern_w_model)
    
    bin_act_model = convert2binary_activation(model)
    testmode!(bin_act_model)
    
    wr = weight_regulizer(model)  
    ar = activation_regulizer(model, train)

    λ1 = update_λ1(λ1min, λ1max, 0, epochs)
    λ2 = update_λ2(λ2min, λ2max, 0, epochs)
    
    if length(history) == 0
        history = (
            smooth_acc=[accuracy(train, model)],
            discrete_acc=[accuracy(train, discrete_model)],
            tern_w_acc=[accuracy(train, tern_w_model)],
            bin_act_acc=[accuracy(train, bin_act_model)],
            loss=[mean_loss(model, train)],
            λ1=[λ1],
            wr=[wr],
            λ2=[λ2],
            ar=[ar],
            periods=[0],
            # mean_gs = []
            )
    end
       
    for period in 1:periods
        for e in 1:epochs


            # ar_list = [0f0]
            trainmode!(model)
            # mean_list = []
            for (x, y) in train
                # TODO new train
                # gs2 = gradient(model -> objective(model, x, y, λ1, λ2, ar_list), model)
                # push!(mean_list, mean(abs.(gs2[1][1][1].batchnorm.β)))
                # gs = gradient(() -> objective(model, x, y, λ1, λ2, ar_list), ps)
                gs = gradient(() -> objective(model, x, y, λ1, λ2), ps)
                Flux.update!(opt, ps, gs)
            end
            # push!(history.mean_gs, mean(mean_list))
            
            testmode!(model)
            acc = accuracy(train, model)
            
            # get accuracies
            # discrete
            discrete_model = convert2discrete(model)
            testmode!(discrete_model)
            discrete_acc = accuracy(train, discrete_model)

            # weights
            tern_w_model = convert2ternary_weights(model)
            testmode!(tern_w_model)
            tern_w_acc = accuracy(train, tern_w_model)

            # get accuracies
            bin_act_model = convert2binary_activation(model)
            testmode!(bin_act_model)
            bin_act_acc = accuracy(train, bin_act_model)
            
            # get regularizations
            wr = weight_regulizer(model)

            ar = activation_regulizer(model, train)
            # ar = sum(ar_list) / (length(ar_list) + 1)

            # update λ
            λ1 = update_λ1(λ1min, λ1max, e, epochs)
            λ2 = update_λ2(λ2min, λ2max, e, epochs)
            
            push!(history.smooth_acc, acc)
            push!(history.discrete_acc, discrete_acc)
            push!(history.tern_w_acc, tern_w_acc)
            push!(history.bin_act_acc, bin_act_acc)
            push!(history.loss, mean_loss(model, train))
            push!(history.λ1, λ1)
            push!(history.wr, wr)
            push!(history.λ2, λ2)
            push!(history.ar, ar)

            # print progress
            showvalues = [
                (:period, period),
                (:epoch, e),
                (:smooth_acc, round(100 * history.smooth_acc[end]; digits=2)),
                (:discrete_acc, round(100 * history.discrete_acc[end]; digits=2)),
                (:ternary_weights_acc, round(100 * history.tern_w_acc[end]; digits=2)),
                (:binary_activation_acc, round(100 * history.bin_act_acc[end]; digits=2)),
                (:loss, round(history.loss[end]; digits=4)),
                (:λ1, λ1),
                (:wr, wr),
                (:λ1reg, wr * λ1),
                (:λ2, λ2),
                (:ar, ar),
                (:λ2reg, ar * λ2),
                # (:mean, history.mean_gs[end]),
            ]
            ProgressMeter.next!(p; showvalues)
        end
    end
    push!(history.periods, length(history.smooth_acc))
    return history
end


function show_all_accuracies(history, show_legend=true)
    f = Figure()
    ax1 = Axis(f[1, 1])

    if :periods in keys(history)
        for period in history.periods
            # lines!(ax1, [period, period], color = :black, linestyle = :dash)
            lines!(ax1, [(period, 0), (period, 1)], color = :black, linestyle = :dash)
        end
end


    lines!(ax1, history.smooth_acc, color = :red, label="smooth acc")
    # axislegend(ax1, "", position = :rb)
    lines!(ax1, history.discrete_acc, color = :orange, label ="discrete acc")
    if :tern_w_acc in keys(history)
        lines!(ax1, history.tern_w_acc, color = :blue, label="ternary weights acc")
    end
    if :bin_act_acc in keys(history)
        lines!(ax1, history.bin_act_acc, color = :purple, label="binary activation acc")
    end

    if show_legend
        axislegend("", position = :rb)
    end
    f
end

function show_regularization(history)
    f = Figure()
    ax1 = Axis(f[1, 1])
    if :periods in keys(history)
        for period in history.periods
            # lines!(ax1, [period, period], color = :black, linestyle = :dash)
            lines!(ax1, [(period, 0), (period, 1)], color = :black, linestyle = :dash)
        end
    end
    
    
    lines!(ax1, history.smooth_acc, color = :red, label="smooth acc", linewidth = 3)
    # axislegend(ax1, "", position = :rb)
    lines!(ax1, history.discrete_acc, color = :orange, label ="discrete acc", linewidth = 3)
    if :tern_w_acc in keys(history)
        lines!(ax1, history.tern_w_acc, color = :deeppink3, label="ternary weights acc", linewidth = 3)
    end
    if :bin_act_acc in keys(history)
        lines!(ax1, history.bin_act_acc, color = :purple, label="binary activation acc", linewidth = 3)
    end
    
    axislegend("accuracies", position = :rb)
    # Legend(f, ax1, framevisible = false)
    # Legend(f, ax2, framevisible = false)

    ax2 = Axis(f[1, 1], yaxisposition = :right)
    hidespines!(ax2)
    hidexdecorations!(ax2)

    lines!(ax2, history.loss, color = :blue, label="loss", linestyle = Linestyle([0.5, 1.0, 1.5, 2.5]), linewidth = 3)
    lines!(ax2, history.wr .* history.λ1, color = :darkblue, label="wr * λ1", linestyle = Linestyle([0.5, 1.0, 1.5, 2.5]), linewidth = 3)
    lines!(ax2, history.ar .* history.λ2, color = :steelblue4, label="ar * λ2", linestyle = Linestyle([0.5, 1.0, 1.5, 2.5]), linewidth = 3)

    axislegend("regularizations", position = :lb)
    
    f
end



function show_ar(history)
    f = Figure()
    ax1 = Axis(f[1, 1])
    
    
    lines!(ax1, history.smooth_acc, color = :red, label="smooth acc", linewidth = 3)
    # axislegend(ax1, "", position = :rb)
    lines!(ax1, history.discrete_acc, color = :orange, label ="discrete acc", linewidth = 3)
    if :tern_w_acc in keys(history)
        lines!(ax1, history.tern_w_acc, color = :deeppink3, label="ternary weights acc", linewidth = 3)
    end
    if :bin_act_acc in keys(history)
        lines!(ax1, history.bin_act_acc, color = :purple, label="binary activation acc", linewidth = 3)
    end
    
    axislegend("accuracies", position = :rb)
    # Legend(f, ax1, framevisible = false)
    # Legend(f, ax2, framevisible = false)

    ax2 = Axis(f[1, 1], yaxisposition = :right)
    hidespines!(ax2)
    hidexdecorations!(ax2)

    lines!(ax2, history.ar, color = :steelblue4, label="ar", linestyle = Linestyle([0.5, 1.0, 1.5, 2.5]), linewidth = 3)

    if :periods in keys(history)
        for period in history.periods
            # lines!(ax1, [period, period], color = :black, linestyle = :dash)
            lines!(ax1, [(period, 0), (period, 1)], color = :black, linestyle = :dash)
        end
    end

    axislegend("regularizations", position = :lb)
    
    f
end

function show_wr(history)
    f = Figure()
    ax1 = Axis(f[1, 1])
    
    
    lines!(ax1, history.smooth_acc, color = :red, label="smooth acc", linewidth = 3)
    # axislegend(ax1, "", position = :rb)
    lines!(ax1, history.discrete_acc, color = :orange, label ="discrete acc", linewidth = 3)
    if :tern_w_acc in keys(history)
        lines!(ax1, history.tern_w_acc, color = :deeppink3, label="ternary weights acc", linewidth = 3)
    end
    if :bin_act_acc in keys(history)
        lines!(ax1, history.bin_act_acc, color = :purple, label="binary activation acc", linewidth = 3)
    end
    
    axislegend("accuracies", position = :rb)
    # Legend(f, ax1, framevisible = false)
    # Legend(f, ax2, framevisible = false)

    ax2 = Axis(f[1, 1], yaxisposition = :right)
    hidespines!(ax2)
    hidexdecorations!(ax2)

    lines!(ax2, history.wr, color = :steelblue4, label="wr", linestyle = Linestyle([0.5, 1.0, 1.5, 2.5]), linewidth = 3)

    if :periods in keys(history)
        for period in history.periods
            # lines!(ax1, [period, period], color = :black, linestyle = :dash)
            lines!(ax1, [(period, 0), (period, 1)], color = :black, linestyle = :dash)
        end
    end

    axislegend("regularizations", position = :lb)
    
    f
end

function show_mean(history)
    f = Figure()
    ax1 = Axis(f[1, 1])
    
    
    lines!(ax1, history.smooth_acc, color = :red, label="smooth acc", linewidth = 3)
    # axislegend(ax1, "", position = :rb)
    lines!(ax1, history.discrete_acc, color = :orange, label ="discrete acc", linewidth = 3)
    if :tern_w_acc in keys(history)
        lines!(ax1, history.tern_w_acc, color = :deeppink3, label="ternary weights acc", linewidth = 3)
    end
    if :bin_act_acc in keys(history)
        lines!(ax1, history.bin_act_acc, color = :purple, label="binary activation acc", linewidth = 3)
    end
    
    axislegend("accuracies", position = :rb)
    # Legend(f, ax1, framevisible = false)
    # Legend(f, ax2, framevisible = false)

    ax2 = Axis(f[1, 1], yaxisposition = :right)
    hidespines!(ax2)
    hidexdecorations!(ax2)

    lines!(ax2, Float64.(history.mean_gs), color = :steelblue4, label="mean", linestyle = Linestyle([0.5, 1.0, 1.5, 2.5]), linewidth = 3)

    if :periods in keys(history)
        for period in history.periods
            # lines!(ax1, [period, period], color = :black, linestyle = :dash)
            lines!(ax1, [(period, 0), (period, 1)], color = :black, linestyle = :dash)
        end
    end

    axislegend("regularizations", position = :lb)
    
    f
end