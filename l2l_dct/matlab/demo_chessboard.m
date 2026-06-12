% Demo chessboard effect — reproduction of Petrovskii fig. 5.15
addpath('..');
img = im2double(imread('../assets/test_image.png'));
if size(img,1) ~= 512
    img = imresize(img, [512 512]);
end

% Fixed-point format S1.10 (12-bit)
fmt = numerictype(1, 12, 10);
img_fp = fi(img, fmt);

fprintf('MATLAB demo: run Python run_phase2_chessboard.py for full pipeline.\n');
fprintf('Export golden vectors via compare_with_python.m\n');
