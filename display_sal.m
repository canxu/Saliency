clear all;
clc;

sal_dir = '/data/Can/NewData/salimap/Seq1/c1';

imglist = dir(fullfile(sal_dir,'*.jpg'));

figure;

for i = 1:length(imglist)
    filename = imglist(i).name;
    im = imread(fullfile(sal_dir,filename));
    imshow(im);
    title(filename);
end