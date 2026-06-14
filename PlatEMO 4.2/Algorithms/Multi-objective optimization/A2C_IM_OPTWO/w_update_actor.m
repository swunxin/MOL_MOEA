function W_new = w_update_actor(A, W_init)
    % Function to update Actor's weight using Advantage feedback
    % Inputs:
    %   A      : Advantage value (scalar)
    %   W_init : Initial weight (scalar)
    %   eta    : Learning rate for weight adjustment
    % Outputs:
    %   W_new  : Updated weight
    % Parameters

    eta = 0.1;    % Learning rate 学习率    
    % Step 1: Update weight using gradient-based update rule
    % Gradient is proportional to the advantage function A
    W_new = W_init + eta * A;

    % Optional: Normalize weight if needed (for probability constraints)
    if W_new < 0
        W_new = 0; % Ensure weight does not go below 0
    end
end