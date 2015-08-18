function display_window(x1,x2,y1,y2)

line([y1,y2],[x1,x1],'Color','r','LineWidth',1);

line([y1,y2],[x2,x2],'Color','r','LineWidth',1);

line([y1,y1],[x1,x2],'Color','r','LineWidth',1);

line([y2,y2],[x1,x2],'Color','r','LineWidth',1);