function imdb = imdb_from_kitty(root_dir, image_set)
% imdb = imdb_from_voc(root_dir, image_set, year)
%   Builds an image database for the PASCAL VOC devkit located
%   at root_dir using the image_set and year.
%
%   Inspired by Andrea Vedaldi's MKL imdb and roidb code.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
 
%imdb.name = 'voc_train_2007'
%imdb.image_dir = '/work4/rbg/VOC2007/VOCdevkit/VOC2007/JPEGImages/'
%imdb.extension = '.jpg'
%imdb.image_ids = {'000001', ... }
%imdb.sizes = [numimages x 2]
%imdb.classes = {'aeroplane', ... }
%imdb.num_classes
%imdb.class_to_id
%imdb.class_ids
%imdb.eval_func = pointer to the function that evaluates detections
%imdb.roidb_func = pointer to the function that returns regions of interest

cache_file = [root_dir '/imdb/cache/imdb_kitty_'  image_set];
try
  load(cache_file);
catch  
  imdb.name = ['kitty_' image_set];
  imdb.image_dir ='/data/Song/song/dataset/song_KITTY/training/image_2';%['/home/song/dataset2/song_KITTY/training/image_2'];
%   imglist = dir(imdb.image_dir);
%   imglist = imglist(3:end);
%   imdb.image_ids = {imglist.name}';
 
  imdb.image_ids = textread('/data/Can/KIITY/image2_ids.txt', '%s');%textread(['/home/song/dataset2/song_KITTY/training/' image_set '.txt'], '%s');
  imdb.extension = 'png';
  imdb.classes = {...
      'Car'
      'Cyclist'
      'Misc'
      'Pedestrian'
      'Person_sitting'
      'Tram'
      'Truck'
      'Van'
      'DontCare'};
  imdb.num_classes = length(imdb.classes);
  imdb.class_to_id = ...
    containers.Map(imdb.classes, 1:imdb.num_classes);
  imdb.class_ids = 1:imdb.num_classes;

  % VOC specific functions for evaluation and region of interest DB
  imdb.image_at = @(i) ...
      sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
 
  for i = 1:length(imdb.image_ids)
    tic_toc_print('imdb (%s): %d/%d\n', imdb.name, i, length(imdb.image_ids));
    info = imfinfo([imdb.image_dir '/' imdb.image_ids{i} '.' imdb.extension]);
    imdb.sizes(i, :) = [info.Height info.Width];
  end

  fprintf('Saving imdb to cache...');
  save(cache_file, 'imdb', '-v7.3');
  fprintf('done\n');
end
