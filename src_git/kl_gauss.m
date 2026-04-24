function D = kl_gauss(m1, S1, m2, S2)
% kl_gauss
% Compute the KL divergence between tow Gaussian distributions:
%  D_KL( N(m1,S1) || N(m2,S2) )
% 
% Haibo Li, School of Mathematics and Statistics, HUST
% 10, April, 2026.
%

S1 = 0.5 * (S1 + S1');
S2 = 0.5 * (S2 + S2');

d = length(m1);

R1 = chol(S1);
R2 = chol(S2);

logdetS1 = 2 * sum(log(diag(R1)));
logdetS2 = 2 * sum(log(diag(R2)));

tr_term = trace(S2 \ S1);
diff = m2 - m1;
quad = diff' * (S2 \ diff);

D = 0.5 * (tr_term - d - (logdetS1 - logdetS2) + quad);

end