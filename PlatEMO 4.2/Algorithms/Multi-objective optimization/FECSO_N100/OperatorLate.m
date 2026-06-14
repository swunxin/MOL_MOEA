function [XD,OV] = OperatorLate(Problem,Loser,W1,W2)
% Late stage: two-winner loser updating
% v_new = r5*v + r6*(xw1 - xl) + r7*(xw2 - xl)
% x_new = xl + v_new
    LD = Loser.decs;
    W1D = W1.decs;
    W2D = W2.decs;
    [N,D] = size(LD);
    LV = Loser.adds(zeros(N,D));
    r5 = repmat(rand(N,1),1,D);
    r6 = repmat(rand(N,1),1,D);
    r7 = repmat(rand(N,1),1,D);
    OV = r5.*LV + r6.*(W1D-LD) + r7.*(W2D-LD);
    XD = LD + OV;
    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    XD = min(max(XD,Lower),Upper);
end
