% Export golden 8-point DCT vector for HLS/Python verification
x = [128 64 32 16 8 4 2 1];
C = dct(eye(8));  % MATLAB orthonormal DCT
y = C * x';
outdir = '../tests/golden';
if ~exist(outdir, 'dir'), mkdir(outdir); end
fid = fopen(fullfile(outdir, 'dct_golden.txt'), 'w');
fprintf(fid, '%.17g\n', y);
fclose(fid);
fprintf('Saved %s\n', fullfile(outdir, 'dct_golden.txt'));
