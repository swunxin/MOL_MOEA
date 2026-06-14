function [OffDec,OffVel] = OperatorEarly(Problem,Loser,Winner)
% Early stage: single-winner loser updating (LMOCSO formula)
% v_new = r3*v + r4*(xw - xl)
% x_new = xl + v_new + r3*(v_new - v)

    LoserDec  = Loser.decs;
    WinnerDec = Winner.decs;
    [N,D]     = size(LoserDec);
    LoserVel  = Loser.adds(zeros(N,D));

    r3 = repmat(rand(N,1),1,D);
    r4 = repmat(rand(N,1),1,D);
    OffVel = r3.*LoserVel + r4.*(WinnerDec-LoserDec);
    OffDec = LoserDec + OffVel + r3.*(OffVel-LoserVel);

    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    OffDec = min(max(OffDec,Lower),Upper);
end
