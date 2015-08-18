close all;
clear all;

addpath(fullfile('/data/Can/song_code/Novateur/matlab'));

dataset = 'New_data';
norm = 1;
modelfilename ='model_KITTY.mat' ;%'model_VOC2007_new.mat';

novateur_test_new(dataset, modelfilename, norm);