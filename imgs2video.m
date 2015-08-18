close all;
clear all;
clc;
imgdir1 = '/data/Can/NewData/Seq2/';
imageNames1 = dir(fullfile(imgdir1,'*.jpg'));
imgdir2 = '/data/Can/NewData/salimap/Seq2/c1';
imageNames2 = dir(fullfile(imgdir2,'*.jpg'));
save_video_dir = '/data/Can/NewData/salimap/';
outputVideo = VideoWriter(fullfile(save_video_dir,'car_highway_cmp.avi'));
outputVideo.FrameRate = 24;%shuttleVideo.FrameRate;
open(outputVideo);
%figure;
for ii = 1:length(imageNames2)
   img1 = rgb2gray(imread(fullfile(imgdir1,imageNames1(ii).name)));
   img2 = imread(fullfile(imgdir2,imageNames2(ii).name));
   %rgbimg2 = img2(:,:,[1,1,1]);
   %img = cat(2,img1,rgbimg2);
  
   %imshow(img);
   img = cat(2,img1,img2);
   img = imresize(img,0.3);
   writeVideo(outputVideo,img);
end

close(outputVideo);