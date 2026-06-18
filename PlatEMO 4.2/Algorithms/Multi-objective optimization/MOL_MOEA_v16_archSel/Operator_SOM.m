function Offspring = Operator_SOM(Problem,Loser,Winner,Peer)
% The competitive swarm optimizer of LMOCSO with random-peer learning

    %% Parameter setting
    LoserDec  = Loser.decs;
    WinnerDec = Winner.decs;
    PeerDec   = Peer.decs;

    [N,D]     = size(LoserDec);
    LoserVel  = Loser.adds(zeros(N,D));
    WinnerVel = Winner.adds(zeros(N,D));

    %% Coefficients 
    r1 = repmat(rand(N,1),1,D);   
    r2 = repmat(rand(N,1),1,D);   
    r3 = 0.5*repmat(rand(N,1),1,D);   

    %% Velocity update:  v = r0*v + r1*(x_w-x) + r2*(x_r-x)
    OffVel = r1.*LoserVel + r2.*(WinnerDec - LoserDec) + r3.*(PeerDec - LoserDec);

    %% Position update:  x = x + v + r3*(v - v_old)
    OffDec = LoserDec + OffVel + r1.*(OffVel - LoserVel);

    %% Add the winners 
    OffDec = [OffDec; WinnerDec];
    OffVel = [OffVel; WinnerVel];

    %% Polynomial mutation 
    Lower  = repmat(Problem.lower,2*N,1);
    Upper  = repmat(Problem.upper,2*N,1);
    disM   = 20;
    Site   = rand(2*N,D) < 1/D;
    mu     = rand(2*N,D);

    OffDec = max(min(OffDec,Upper),Lower);

    temp   = Site & mu<=0.5;
    OffDec(temp) = OffDec(temp) + (Upper(temp)-Lower(temp)).* ...
        ((2.*mu(temp)+(1-2.*mu(temp)).*(1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)) - 1);

    temp   = Site & mu>0.5;
    OffDec(temp) = OffDec(temp) + (Upper(temp)-Lower(temp)).* ...
        (1 - (2.*(1-mu(temp)) + 2.*(mu(temp)-0.5).*(1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));

    Offspring = Problem.Evaluation(OffDec,OffVel);
end
