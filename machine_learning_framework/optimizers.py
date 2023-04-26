def gradient_descent(weights, biases, dW, db, alpha, lambda_, m):
    n_layers = len(weights)
    for i in range(n_layers):
        regularization = (lambda_/ m) * weights[i]
        weights[i] -= alpha * (dW[i] + regularization) 
        biases[i] -= alpha * db[i]

    return weights, biases