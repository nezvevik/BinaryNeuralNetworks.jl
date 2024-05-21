using BTNNs: BTLayer, RLayer, accuracy, get_ternary_quantizer, set_weight_quantizer
using BTNNs: weight_regularizer, activation_regularizer, get_binary_regularizer, set_output_regularizer, get_ternary_regularizer
using BTNNs: convert2discrete, convert2ternary_weights, convert2binary_activation
using Flux: Chain, AdaBelief, mean, mse, logitcrossentropy, sum, params, gradient, update!, trainmode!, testmode!, accuracy
using ProgressMeter

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

λ1min, λ1max = get_λ_range(wr, l)
λ2min, λ2max = get_λ_range(ar, l)

history = []
periods = 1
epochs = 10

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

function train_reg!(model, opt, train, objective;
    history = [],
    update_λ1=update_λ1, update_λ2=update_λ2,
    weight_regularizer=weight_regularizer, activation_regularizer=activation_regularizer,
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
    
    wr = weight_regularizer(model)  
    ar = activation_regularizer(model, train)

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
                update!(opt, ps, gs)
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
            wr = weight_regularizer(model)

            ar = activation_regularizer(model, train)
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

function train_reg_no_log!(model, opt, train, objective;
    history = [],
    update_λ1=update_λ1, update_λ2=update_λ2,
    weight_regularizer=weight_regularizer, activation_regularizer=activation_regularizer,
    periods::Int=4, epochs::Int=25,
    λ1min=10e-6, λ1max=10e-3,
    λ2min=10e-6, λ2max=10e-3)
    
    discrete_model = convert2discrete(model)
    testmode!(discrete_model)

    p = Progress(periods*epochs, 1)
    ps = params(model)
    
    λ1 = update_λ1(λ1min, λ1max, 0, epochs)
    λ2 = update_λ2(λ2min, λ2max, 0, epochs)
    
    if length(history) == 0
        history = (
            smooth_acc=[accuracy(train, model)],
            discrete_acc=[accuracy(train, discrete_model)],
            loss=[mean_loss(model, train)],
            λ1=[λ1],
            λ2=[λ2],
            periods=[0],
            )
    end
       
    for period in 1:periods
        for e in 1:epochs
            trainmode!(model)
            for (x, y) in train
                gs = gradient(() -> objective(model, x, y, λ1, λ2), ps)
                update!(opt, ps, gs)
            end
            
            testmode!(model)
            acc = accuracy(train, model)

            discrete_model = convert2discrete(model)
            testmode!(discrete_model)
            discrete_acc = accuracy(train, discrete_model)
    

            # update λ
            λ1 = update_λ1(λ1min, λ1max, e, epochs)
            λ2 = update_λ2(λ2min, λ2max, e, epochs)
            
            push!(history.smooth_acc, acc)
            push!(history.discrete_acc, discrete_acc)
            push!(history.loss, mean_loss(model, train))
            push!(history.λ1, λ1)
            push!(history.λ2, λ2)

            # print progress
            showvalues = [
                (:period, period),
                (:epoch, e),
                (:smooth_acc, round(100 * history.smooth_acc[end]; digits=2)),
                (:discrete_acc, round(100 * history.discrete_acc[end]; digits=2)),
                (:loss, round(history.loss[end]; digits=4)),
                (:λ1, λ1),
                (:λ2, λ2),
            ]
            ProgressMeter.next!(p; showvalues)
        end
    end
    push!(history.periods, length(history.smooth_acc))
    return history
end


T1 = [-0.5, -0.45, -0.43, -0.42, -0.4,-0.38, -0.35]
T2 = [0.3, 0.35, 0.4, 0.42, 0.44, 0.45, 0.46, 0.5]


best_t1, best_t2, best_acc = 0, 0, 0

best_t1, best_t2, best_acc

for t1 in T1
    for t2 in T2
        if t1 >= t2 continue end
        set_weight_quantizer(model, get_ternary_quantizer(t1, t2))
        discrete = convert2discrete(model)
        acc = accuracy(train_data, discrete)
        if acc > best_acc
            best_acc = acc
            best_t1 = t1
            best_t2 = t2
        end
        println(acc)
    end
end

best_acc

best_t1, best_t2