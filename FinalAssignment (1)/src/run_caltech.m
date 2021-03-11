%clear all
%close all

run('vlfeat-0.9.21/toolbox/vl_setup.m');
tic
acc = phow_caltech101();
fprintf('Caltech101 test accuracy = %0.2f%%\n', acc);
toc
