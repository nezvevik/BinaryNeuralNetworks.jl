using BTNNs: BTLayer, train_st!, STLayer, PSALayer, RLayer, accuracy, ternary_quantizer, set_weight_quantizer, convert2discrete
using BTNNs: weight_regularizer, activation_regularizer, set_output_regularizer, get_binary_regularizer, set_weight_regularizer
using Flux: Chain, AdaBelief, mean, mse, logitcrossentropy, sum, params, gradient, update!, trainmode!, testmode!
using Flux: Adam, Dense
using ProgressMeter

using DelimitedFiles: writedlm

train_data, test_data = createloader_MNIST()
k=10
# DENSE
model = Chain(
    Dense(784, 256, tanh),
    Dense(256, 100, tanh),
    Dense(100, 10, tanh),
)

model = Chain(
    Dense(13*k, 50, tanh),
    Dense(50, 3, tanh),
)

model = Chain(
    Dense(input_size, 20, tanh),
    Dense(20, nclasses, tanh),
)

# STE
model_ste = Chain(
    STLayer(784, 256, tanh),
    STLayer(256, 100, tanh),
    STLayer(100, 10, tanh),
)

model_ste = Chain(
    STLayer(13*k, 50, tanh),
    STLayer(50, 3, tanh),
)
model
accuracy(train_data, model)
accuracy(test_data, model)
history = train_st!(model, AdaBelief(), train_data, test_data, epochs=400)

include("../data/datasets.jl")

train_data, test_data = createloader_Flower(2000, 900, k)
input_size = size(first(train_data)[1], 1)
nclasses = size(first(train_data)[2], 1)

model_ste = Chain(
    STLayer(input_size, 20, tanh),
    STLayer(20, nclasses, tanh),
)

accuracy(train_data, model_ste)
accuracy(test_data, model_ste)

history_ste = train_st!(model_ste, AdaBelief(), train_data, test_data, epochs=100)
writedlm("history/history_ste_flower.csv", history_ste)


# PSA
train_data, test_data = createloader_Wine(k, 0.9)
model_psa = Chain(
    PSALayer(784, 256, identity),
    PSALayer(256, 100, identity),
    PSALayer(100, 10, identity),
)
model_psa = Chain(
    PSALayer(13*k, 50, identity),
    PSALayer(50, 3, identity),
)
train_data, test_data = createloader_Flower(2000, 900, k)
input_size = size(first(train_data)[1], 1)
nclasses = size(first(train_data)[2], 1)
model_psa = Chain(
    PSALayer(input_size, 20, identity),
    PSALayer(20, nclasses, identity),
)
    
    
    
accuracy(train_data, model_psa)
accuracy(test_data, model_psa)
    
    
history_psa = train_st!(model_psa, AdaBelief(), train_data, test_data, epochs=1000)
writedlm("history/history_psa_flower.csv", history_psa)

history_psa

# REG
model_reg = Chain(
    RLayer(28*28, 256, tanh),
    RLayer(256, 100, tanh),
    RLayer(100, 10, tanh)
)

model_reg = Chain(
    RLayer(13*k, 50, tanh),
    RLayer(50, 3, tanh)
)


history = []
periods = 1
epochs = 100
loss=logitcrossentropy

train_data, test_data = createloader_Wine(k, 0.9)

accuracy(train_data, model_reg)


function objective(m, x, y, λ1, λ2)
    o, r = m((x, 0f0))
    loss(o, y) + λ1 * weight_regularizer(m) + λ2 * r
end


history = train_reg!(model_reg, AdaBelief(), train_data, objective;
    history=history, periods=periods, epochs=epochs)

writedlm("history/history_reg.csv", history)


function mean_loss(m, data)
    mean(loss(m(x), y) for (x, y) in data)
end
update_λ1(λ1min, λ1max, i, epochs) = λ1min + 1/2 * (λ1max - λ1min) * (1 + cos(i/(epochs) * π))
update_λ1(λ1min, λ1max, i, epochs) = 0f0
update_λ2(λ2min, λ2max, i, epochs) = λ2min + 1/2 * (λ2max - λ2min) * (1 + cos(i/(epochs) * π))

function get_λ_range(reg, l)
    return 0f0, l / reg
end
function train_reg!(model, opt, train, test, objective;
    history = [],
    update_λ1=update_λ1, update_λ2=update_λ2,
    weight_regularizer=weight_regularizer, activation_regularizer=activation_regularizer,
    periods::Int=4, epochs::Int=25,)
    
    discrete_model = convert2discrete(model)
    testmode!(discrete_model)

    p = Progress(periods*epochs, 1)
    ps = params(model)
    
    
    if length(history) == 0
        history = (
            smooth_acc=[accuracy(train, model)],
            smooth_test_acc=[accuracy(test, model)],
            discrete_acc=[accuracy(train, discrete_model)],
            discrete_test_acc=[accuracy(test, discrete_model)],
            loss=[mean_loss(model, train)],
            λ1=[0f0],
            λ2=[0f0],
            periods=[0],
            )
    end
       
    for period in 1:periods
        l = mean_loss(model, train_data)

        # set λs 
        wr = weight_regularizer(model)
        ar = activation_regularizer(model, train_data)

        λ1min, λ1max = get_λ_range(wr, l)
        λ2min, λ2max = get_λ_range(ar, l)

        λ1 = 0f0
        λ2 = 0f0


        for e in 1:epochs
            trainmode!(model)
            for (x, y) in train
                gs = gradient(() -> objective(model, x, y, λ1, λ2), ps)
                update!(opt, ps, gs)
            end
            
            testmode!(model)
            acc = accuracy(train, model)
            test_acc = accuracy(test, model)

            discrete_model = convert2discrete(model)
            testmode!(discrete_model)
            discrete_acc = accuracy(train, discrete_model)
            discrete_test_acc = accuracy(test, discrete_model)


    

            # update λ
            if period == 1
                λ1 = 0f0
                λ2 = 0f0
            elseif period == 2
                λ1 = 0f0
                λ2 = update_λ2(λ2min, λ2max, e, epochs)
            else
                λ1 = update_λ1(λ1min, λ1max, e, epochs)
                λ2 = update_λ2(λ2min, λ2max, e, epochs)
            end
            
            push!(history.smooth_acc, acc)
            push!(history.smooth_test_acc, test_acc)
            push!(history.discrete_acc, discrete_acc)
            push!(history.discrete_test_acc, discrete_test_acc)
            push!(history.loss, mean_loss(model, train))
            push!(history.λ1, λ1)
            push!(history.λ2, λ2)

            # print progress
            showvalues = [
                (:period, period),
                (:epoch, e),
                (:smooth_acc, round(100 * history.smooth_acc[end]; digits=2)),
                (:discrete_acc, round(100 * history.discrete_acc[end]; digits=2)),
                (:discrete_test_acc, round(100 * history.discrete_test_acc[end]; digits=2)),
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