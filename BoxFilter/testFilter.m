% Main function for your assignment
% read the output number and intput your answer to your assignment!

im = imread('Lenna.png');
im = rgb2gray(im);
im = im2double(im);

% first trial with 3x3
out1 = BoxFilter(im, 3);
%figure, imshow(out1);
out2 = BoxFilter(im, 11);
%figure, imshow(out2);

diff = abs(out1-out2);
% answer!!!! The number is the answer.
fprintf('Ans: %0.2f\n', sum(sum(diff)));


