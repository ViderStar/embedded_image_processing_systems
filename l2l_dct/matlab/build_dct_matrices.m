% Build DCT-II / DCT-III matrices (reference for Python cross-check)
N = 8;
C = zeros(N, N);
for k = 0:N-1
    for n = 0:N-1
        if k == 0
            scale = 1 / sqrt(N);
        else
            scale = sqrt(2 / N);
        end
        C(k+1, n+1) = scale * cos(pi * k * (2*n + 1) / (2*N));
    end
end
D = C';
fprintf('Orthogonality error ||C*C'' - I|| = %.2e\n', norm(C*C' - eye(N)));
save('golden/dct_matrices.mat', 'C', 'D');
