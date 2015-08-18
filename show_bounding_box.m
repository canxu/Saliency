clear all;
clc;
addpath('/home/cax001/Detection_proposal/devkit/matlab');
img_dir = '/data/Song/song/dataset/song_KITTY/training/image_2';
label_dir = '/data/Song/song/dataset/song_KITTY/training/label_2';
fea_dir = '/data/Can/KIITY/image2_training_fea_conv8';
root_dir = pwd;
cache_dir = fullfile(root_dir,'imdb/cache');

imdb = imdb_from_kitty(root_dir, 'image_2');

for idx = 1:5
    img_id = imdb.image_ids{idx};
    img_name = fullfile(imdb.image_dir,[img_id '.' imdb.extension]);
    im = imread(img_name);
    objects = readLabels(label_dir,str2num(img_id));
    figure;
    imshow(im);
    im_size = imdb.sizes(idx,:);
    for i = 1:length(objects)
        obj = objects(i);
        x1 = max(round(obj.x1),1);
        y1 = max(round(obj.y1),1);
        x2 = min(round(obj.x2),im_size(2));
        y2 = min(round(obj.y2),im_size(1));
        display_window(y1,y2,x1,x2);
    end
end