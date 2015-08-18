function feat = extract_cnn(im, featname, type)
switch type
    case 'conv5'
        temp = load([featname]);
        temp = temp.CNN_feature; 
        featsize = size(temp); 
        if(length(featsize)==4)
            featsize = featsize(2:4);
        end 
        temp = reshape(temp, [featsize(1) featsize(2)*featsize(3)]);
        feat.feat = double(temp); 
        feat.wid = size(im,2);
        feat.hgt = size(im,1);
        pady = round( (feat.hgt - ((featsize(2) - 1)*16+1))/2 );
        padx = round( (feat.wid - ((featsize(3) - 1)*16+1))/2 );
        indy = pady + 1 + (0:(featsize(2) - 1))*16;
        indx = padx + 1 + (0:(featsize(3) - 1))*16;
        [X, Y] = meshgrid(indx, indy);
        feat.y = Y(:)';
        feat.x = X(:)';
end  