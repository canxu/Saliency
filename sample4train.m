clear all;
clc;
dataset = 'KITTY_image2';
addpath('/home/cax001/Detection_proposal/devkit/matlab');
img_dir = '/data/Song/song/dataset/song_KITTY/training/image_2';
label_dir = '/data/Song/song/dataset/song_KITTY/training/label_2';
fea_dir = '/data/Can/KIITY/image2_training_fea_conv8';
root_dir = pwd;
cache_dir = fullfile(root_dir,'imdb/cache');

imdb = imdb_from_kitty(root_dir, 'image_2');

short_edge = 512;

img_ids = imdb.image_ids;
sample_num = 130000;
fea_dim = 1000;
%pos_num = zeros(1,imdb.num_classes);
load pos_num.mat;%training feature nums with -s 512

for cls_idx = 1:imdb.num_classes
    if(pos_num(cls_idx)>sample_num)
        fea_pos = zeros(fea_dim,sample_num);
    else
        fea_pos = zeros(fea_dim,pos_num(cls_idx));
    end
   % fea_pos = zeros(fea_dim,sample_num);
    cnt = 0;
    for img_idx = 1:length(img_ids)
        im_name = img_ids{img_idx};
        im_size = imdb.sizes(img_idx,:);
        fea_name = [im_name '_CNN_fc8-conv_feature.mat'];
        load(fullfile(fea_dir,fea_name));
        fea_size = size(CNN_feature);
        fea_size = fea_size(3:4);
        CNN_feature = reshape(CNN_feature, [fea_dim fea_size(1)*fea_size(2)]);
        [x0,y0] = meshgrid(1:fea_size(2),1:fea_size(1));
        
        x = (x0-1)*32+112;
        y = (y0-1)*32+112;
        
        height = im_size(1);
        wid = im_size(2);
        
        if(height<wid)
            rescale = height*1.0/short_edge;
        else
            rescale = wid*1.0/short_edge;
        end
        
        x = x*rescale;
        y = y*rescale;
        
        x = x(:);
        y = y(:);
        fea_idx = [];
        objects = readLabels(label_dir,str2num(im_name));
        for k = 1:length(objects)
            obj = objects(k);
            if(imdb.class_to_id(obj.type)==cls_idx)
                x1 = max(round(obj.x1),1);
                y1 = max(round(obj.y1),1);
                x2 = min(round(obj.x2),im_size(2));
                y2 = min(round(obj.y2),im_size(1));
                fea_idx = [fea_idx;find(x>x1 & x<x2 & y>y1 & y<y2)];
            end
        end
        fea_idx = unique(fea_idx);
        
        if(pos_num(cls_idx)>sample_num)
            actual_sample_num = floor(sample_num*1.0/pos_num(cls_idx)*length(fea_idx));
            perm = randperm(length(fea_idx));
            sample_idx = fea_idx(perm(1:actual_sample_num));
            fea_pos(:,cnt+1:cnt+actual_sample_num) = CNN_feature(:,sample_idx);
            cnt = cnt+actual_sample_num;
        else
            fea_pos(:,cnt+1:cnt+length(fea_idx)) = CNN_feature(:,fea_idx);
            cnt = cnt+length(fea_idx);
        end
       
         %pos_num(cls_idx) = pos_num(cls_idx)+length(fea_idx);
    end
    
    fea_pos = fea_pos(:,1:cnt);
    save_name = [dataset '_pos' num2str(cls_idx)];
    save(fullfile(cache_dir,save_name),'fea_pos');
    clear fea_pos;
    
end
%pos_num
%save('pos_num.mat','pos_num');
%generate neg_feature
%}

neg_per_image = round(sample_num/length(imdb.image_ids));
neg_num = neg_per_image*length(imdb.image_ids);
fea_neg = zeros(fea_dim,neg_num);

%for cls_idx = 1:imdb.num_classes
  %  fea_pos = zeros(fea_dim,sample_num);
    cnt = 0;
    for img_idx = 1:length(img_ids)
     im_name = img_ids{img_idx};
     im_size = imdb.sizes(img_idx,:);
     fea_name = [im_name '_CNN_fc8-conv_feature.mat'];
     load(fullfile(fea_dir,fea_name));
     fea_size = size(CNN_feature);
     fea_size = fea_size(3:4);
     CNN_feature = reshape(CNN_feature, [fea_dim fea_size(1)*fea_size(2)]);
    [x0,y0] = meshgrid(1:fea_size(2),1:fea_size(1));
    
     x = (x0-1)*32+112;
     y = (y0-1)*32+112;
     
     height = im_size(1);
     wid = im_size(2);
     
     if(height<wid)
         rescale = height*1.0/short_edge;
     else
         rescale = wid*1.0/short_edge;
     end
     
     x = x*rescale;
     y = y*rescale;
     
     x = x(:);
     y = y(:);
     fea_idx = [];
     objects = readLabels(label_dir,str2num(im_name));
     for k = 1:length(objects)
        obj = objects(k);
     
            x1 = max(round(obj.x1),1);
            y1 = max(round(obj.y1),1);
            x2 = min(round(obj.x2),im_size(2));
            y2 = min(round(obj.y2),im_size(1));
            fea_idx = [fea_idx;find(x>x1 & x<x2 & y>y1 & y<y2)];
      
     end
     ind = unique(fea_idx);
     ind2 = setdiff(1:fea_size(1)*fea_size(2),ind);
     perm2 = randperm(length(ind2));
     fea_neg(:,cnt+1:cnt+neg_per_image) = CNN_feature(:,ind2(perm2(1:neg_per_image)));
     cnt = cnt+neg_per_image;

    end
    
  
    save_name = [dataset '_neg'];
    save(fullfile(cache_dir,save_name),'fea_neg');
    clear fea_neg;
    

%}