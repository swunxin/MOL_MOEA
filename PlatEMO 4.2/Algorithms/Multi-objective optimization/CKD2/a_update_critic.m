function [V_new, A] = a_update_critic(rt, rt_mean, V_old, trainingGapCounter)
    % Compute V_target and Advantage A for a single step
    % Inputs:
    %   rt      : Single reward value at current step
    %   rt_mean : Mean of rewards (used as V_next approximation)
    %   V_old   : Previous state value (V(s_t))
    % Outputs:
    %   V_new   : Updated state value
    %   A       : Advantage for current step
    
    % Parameters
    gamma = 0.2;  % Discount factor
    alpha = 0.8;  % EMA smoothing factor
    
    % Compute V_next using rt_mean
    if trainingGapCounter <= 3
        V_next = rt_mean;
    else
        V_next = V_old;
    end 
    % Compute V_target  0.2*V_old + rt_mean*0.8;
    V_target = rt + gamma * V_next;  % Temporal Difference target
    
    % Compute Advantage
    A = V_target - V_next;  % Advantage function: difference between target and next value
    
    % Update V(s_t) using EMA
    V_new = alpha * V_target + (1 - alpha) * V_old;  % Exponential Moving Average update
end