% Fixed-point ladder DCT-ODCT demo (S1.10, 12-bit)
% Cross-check with Python: run_phase2_chessboard.py

fmt = numerictype(1, 12, 10);
img = im2double(imread('../assets/test_image.png'));
if size(img,1) ~= 512
    img = imresize(img, [512 512]);
end

N = 8;
C = zeros(N);
for k = 0:N-1
    for n = 0:N-1
        if k == 0, sc = 1/sqrt(N); else, sc = sqrt(2/N); end
        C(k+1,n+1) = sc * cos(pi*k*(2*n+1)/(2*N));
    end
end
D = C';

block = img(1:N, 1:N);
block_fp = fi(block, fmt);
coeff = C * double(block_fp) * C';
coeff_q = fi(coeff, fmt);
recon = D * double(coeff_q) * D';

fprintf('Block max recon error (quantized, no SIB): %.6f\n', max(abs(double(block_fp(:) - recon(:)))));
