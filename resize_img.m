clear all;
clc;
dataset = 'KITTY';
addpath(fullfile('/home/cax001/Detection_proposal/devkit/matlab'));
datasetpath = '/home/song/dataset2/song_KITTY/training';
imgpath = [datasetpath '/image_2'];
labelpath = '/home/song/dataset2/song_KITTY/training/label_2';
image_ids = textread([datasetpath '/data_new/sublist/val.txt'], '%s');
save_prefix = '/data/Can/img_resized';
save_prefix =fullfile(save_prefix,dataset);
if(~exist(save_prefix,'dir'))
    mkdir(save_prefix);
end
save_prefix = fullfile(save_prefix,'training');
if(~exist(save_prefix,'dir'))
    mkdir(save_prefix);
end
save_prefix = fullfile(save_prefix,'image_2');

if(~exist(save_prefix,'dir'))
    mkdir(save_prefix);
end
scale = 2;
save_prefix = fullfile(save_prefix,['scale' num2str(scale)]);
mkdir(save_prefix);
for idx =1:length(image_ids)
    idx
    img_idx = str2num(image_ids{idx});
    %object = readLabels(labelpath,img_idx);
    
    im = imread(sprintf('%s/%06d.png',imgpath,img_idx));
    new_im = imresize(im,scale);
    save_path = sprintf('%s/%06d.png',save_prefix,img_idx);
    imwrite(new_im,save_path);
end