clearvars
close

% creating data by generating Multivariate Normal Random Numbers
mean = [5.0, 6.0];
covariance = [1.0 0.95; 0.95 1.2];
rng('default')  % For reproducibility
data = mvnrnd(mean, covariance, 8000);
 
% visualising data
% scatter(data(:,1),data(:,2),'.');

% detrimine split factor
shape = size(data);
data = [ones(shape(1), 1), data];
shape = size(data);
split_factor = 0.90;
split = fix(split_factor * shape(1));

% training data
X_train = data(1:split, 1:shape(2)-1);
y_train = data(1:split, shape(2));
%y_train = y_train.';

% test data 
X_test = data(split+1:shape(1), 1:shape(2)-1);
y_test = data(split+1:shape(1), shape(2));
%y_test = y_test.';

sz_train = size(X_train);
sz_test = size(X_test);

sprintf("Number of examples in dataset is: %d", shape(1))
sprintf("Number of examples in training set is: %d", sz_train(1))
sprintf("Number of examples in testing set is: %d", sz_test(1))

learning_rate =  0.001;
batch_size = 32;

[theta, error_list] = gradientDescent(X_train, y_train, learning_rate, batch_size);

[t1, t2] = size(theta);
sprintf("Bias = %d", theta(1))
sprintf("Coefficients = %d", theta(2:t1))

%predicting output for X_test
y_pred = hypothesis(X_test, theta);

% calculating error in predictions
[test_sz1, test_sz2] = size(y_test);
error = sum(abs(y_test - y_pred) / test_sz1, 'all');
sprintf("Mean absolute error = %d", error)

% visualising data
figure();
subplot(2,2,1);
scatter(X_train(:,2),y_train,'.');
title('Training dataset');

subplot(2,2,2);
plot(error_list);
xlabel("Iterations")
ylabel("Cost")
title('Cost function');

subplot(2,2,3);
scatter(X_test(:, 2), y_test,'.');
title('Test dataset');

subplot(2,2,4);
plot(X_test(:, 2), y_test,'.', X_test(:, 2), y_pred, 'm.');
title('Predictions');

% linear regression using "mini-batch" gradient descent
% function to compute hypothesis / predictions
function  hyp = hypothesis(X, theta)
hyp = X*theta;
end

% function to compute gradient of error function w.r.t. theta
function grad = gradient(X, y, theta)
    h = hypothesis(X, theta);
    transpose_X = X.';
    grad = transpose_X*(h - y);
end

%function to compute the error for current values of theta
function c = cost(X, y, theta)
    h = hypothesis(X, theta);
    J = ((h - y).')*(h - y);
    J = J / 2;
    c = J(1);
end

%function to create a list containing mini-batches
function mini_batches = create_mini_batches(X, y, batch_size)
    mini_batches = {};
    data = [X, y];
    %np.random.shuffle(data);
    [rows, cols] = size(data);
    n = fix(rows/batch_size);
    index_end = 0;
    for i=1:n
        index_start = index_end + 1;
        index_end  = i*batch_size;
        mini_batch = data(index_start:index_end, :);
        X_mini = mini_batch(:, 1:cols-1);
        Y_mini = mini_batch(:, cols);
        mini_batches{end+1} = {X_mini, Y_mini};
    end
    if mod(rows, batch_size) ~= 0
        mini_batch = data(i * batch_size:rows);
        X_mini = mini_batch(:, 1:cols-1);
        Y_mini = mini_batch(:, cols);
        mini_batches{end+1} = {X_mini, Y_mini};
    end
end

% function to perform mini-batch gradient descent
function [theta, errors] = gradientDescent(X, y, learning_rate, batch_size)
    X_size = size(X);
    theta = zeros(X_size(2), 1);
    errors = [];
    max_iters = 3;
    for itr=1:max_iters
        mini_batches = create_mini_batches(X, y, batch_size);
        for mini_batch = mini_batches
            cell = mini_batch{1};
            X_mini = cell{1};
            y_mini = cell{2};
            theta = theta - learning_rate * gradient(X_mini, y_mini, theta);
            errors(end+1)=(cost(X_mini, y_mini, theta));
        end
    end
end