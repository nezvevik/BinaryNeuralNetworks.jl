using BTNNs: BTLayer, RLayer, accuracy, ternary_quantizer, set_weight_quantizer, convert2discrete
using BTNNs: weight_regularizer, activation_regularizer, set_output_regularizer, get_binary_regularizer, set_weight_regularizer
using Flux: Chain, AdaBelief, mean, mse, logitcrossentropy, sum, params, gradient, update!, trainmode!, testmode!
using Flux: Adam
using ProgressMeter


include("../data/datasets.jl")

loss=logitcrossentropy

function mean_loss(m, data)
    mean(loss(m(x), y) for (x, y) in data)
end

function objective(m, x, y, λ1, λ2)
    o, r = m((x, 0f0))
    loss(o, y) + λ1 * weight_regularizer(m) + λ2 * r
end

function get_λ_range(reg, l)
    return 0f0, l / reg
end


train_data, test_data = createloader_MNIST()

optims = [Adam(0.01), Adam(0.001), AdaBelief(0.01), AdaBelief(0.001), AdaBelief(0.0001)]

w_reg_power = [0.5, 1., 2., 4., 8.]
a_reg_power = [0.5, 1., 2., 4., 8.]




model = Chain(
    RLayer(28*28, 100, tanh),
    RLayer(100, 10, tanh)
)


wrp = rand(w_reg_power)
arp = rand(a_reg_power)
optim = rand(optims)

set_weight_regularizer(model, get_ternary_regularizer(wrp))
set_output_regularizer(model, get_binary_regularizer(arp))
 

config = Dict(
    :wrp => wrp,
    :arp => arp,
    :optim => optim,
)

update_λ1(λ1min, λ1max, i, epochs) = λ1min + 1/2 * (λ1max - λ1min) * (1 + cos(i/(epochs) * π))
update_λ2(λ2min, λ2max, i, epochs) = λ2min + 1/2 * (λ2max - λ2min) * (1 + cos(i/(epochs) * π))
history = []
periods = 3
epochs = 7

history = train_reg!(model, optim, train_data, objective;
    history=history, periods=periods, epochs=epochs)


function train_reg!(model, opt, train, objective;
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
            discrete_acc=[accuracy(train, discrete_model)],
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

            discrete_model = convert2discrete(model)
            testmode!(discrete_model)
            discrete_acc = accuracy(train, discrete_model)
    

            # update λ
            if period == 1
                λ1 = 0f0
                λ2 = 0f0
            elseif period == 2
                λ1 = update_λ1(λ1min, λ1max, e, epochs)
                λ2 = update_λ2(λ2min, λ2max, e, epochs)
            else
                λ1 = update_λ1(λ1min, λ1max, e, epochs) * 8
                λ2 = update_λ2(λ2min, λ2max, e, epochs) * 8
            end
            
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





