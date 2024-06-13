using Flux: AdaBelief, mean, mse, logitcrossentropy, sum, params, gradient, update!, trainmode!, testmode!
using ProgressMeter
using BTNNs: convert2discrete, convert2binary_activation, convert2ternary_weights
using BTNNs: activation_regularizer, weight_regularizer, error, accuracy

function train_default!(model, opt, train, test; loss=logitcrossentropy, history = [],epochs=30)
    function mean_loss(m, data)
        mean(loss(m(x), y) for (x, y) in data)
    end
    discrete_model = convert2discrete(model)
    testmode!(discrete_model)

    p = Progress(epochs, 1)
    ps = params(model)
    
    
    if length(history) == 0
        history = (
            smooth_acc=[accuracy(train, model)],
            smooth_test_acc=[accuracy(test, model)],
            discrete_test_acc=[accuracy(test, discrete_model)],
            loss=[mean_loss(model, train)],
            )
    end
       
    for e in 1:epochs
        trainmode!(model)
        for (x, y) in train
            gs = gradient(() -> loss(model(x), y), ps)
            update!(opt, ps, gs)
        end
        
        testmode!(model)
        acc = accuracy(train, model)

        test_acc = accuracy(test, model)

        discrete_model = convert2discrete(model)
        testmode!(discrete_model)
        discrete_test_acc = accuracy(test, discrete_model)


        
        push!(history.smooth_acc, acc)
        push!(history.smooth_test_acc, test_acc)
        push!(history.discrete_test_acc, discrete_test_acc)
        push!(history.loss, mean_loss(model, train))

        # print progress
        showvalues = [
            (:epoch, e),
            (:smooth_acc, round(100 * history.smooth_acc[end]; digits=2)),
            (:smooth_test_acc, round(100 * history.smooth_test_acc[end]; digits=2)),
            (:discrete_test_acc, round(100 * history.discrete_test_acc[end]; digits=2)),
            (:loss, round(history.loss[end]; digits=4)),
        ]
        ProgressMeter.next!(p; showvalues)
    end
    return history
end

function train_st!(model, opt, train, test; loss=logitcrossentropy, history = [],epochs=30)
    function mean_loss(m, data)
        mean(loss(m(x), y) for (x, y) in data)
    end

    p = Progress(epochs, 1)
    ps = params(model)
    
    
    if length(history) == 0
        history = (
            smooth_acc=[accuracy(train, model)],
            smooth_test_acc=[accuracy(test, model)],
            loss=[mean_loss(model, train)],
            )
    end
       
    for e in 1:epochs
        trainmode!(model)
        for (x, y) in train
            gs = gradient(() -> loss(model(x), y), ps)
            update!(opt, ps, gs)
        end
        
        testmode!(model)
        acc = accuracy(train, model)

        test_acc = accuracy(test, model)

        
        push!(history.smooth_acc, acc)
        push!(history.smooth_test_acc, test_acc)
        push!(history.loss, mean_loss(model, train))

        # print progress
        showvalues = [
            (:epoch, e),
            (:smooth_acc, round(100 * history.smooth_acc[end]; digits=2)),
            (:smooth_test_acc, round(100 * history.smooth_test_acc[end]; digits=2)),
            (:loss, round(history.loss[end]; digits=4)),
        ]
        ProgressMeter.next!(p; showvalues)
    end
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
            push!(history.λ1, λ1)
            push!(history.λ2, λ2)

            # print progress
            showvalues = [
                (:period, period),
                (:epoch, e),
                (:smooth_acc, round(100 * history.smooth_acc[end]; digits=2)),
                (:discrete_acc, round(100 * history.discrete_acc[end]; digits=2)),
                (:λ1, λ1),
                (:λ2, λ2),
            ]
            ProgressMeter.next!(p; showvalues)
        end
    end
    push!(history.periods, length(history.smooth_acc))
    return history
end

