classdef MTXXmlapp < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure       matlab.ui.Figure
        Slider_31      matlab.ui.control.Slider
        Label_23       matlab.ui.control.Label
        Slider_30      matlab.ui.control.Slider
        Label_22       matlab.ui.control.Label
        Button_13      matlab.ui.control.StateButton
        Slider_29      matlab.ui.control.Slider
        Label_21       matlab.ui.control.Label
        Slider_28      matlab.ui.control.Slider
        Label_20       matlab.ui.control.Label
        Slider_25      matlab.ui.control.Slider
        Label_19       matlab.ui.control.Label
        Button_12      matlab.ui.control.StateButton
        Slider_21      matlab.ui.control.Slider
        Label_18       matlab.ui.control.Label
        Slider_20      matlab.ui.control.Slider
        Label_17       matlab.ui.control.Label
        Slider_19      matlab.ui.control.Slider
        Label_16       matlab.ui.control.Label
        Slider_18      matlab.ui.control.Slider
        Label_15       matlab.ui.control.Label
        Slider_16      matlab.ui.control.Slider
        Label_13       matlab.ui.control.Label
        Slider_14      matlab.ui.control.Slider
        Label_10       matlab.ui.control.Label
        Slider_13      matlab.ui.control.Slider
        Label_12       matlab.ui.control.Label
        Button_11      matlab.ui.control.StateButton
        Slider_6       matlab.ui.control.Slider
        Label_5        matlab.ui.control.Label
        Slider_5       matlab.ui.control.Slider
        Slider_5Label  matlab.ui.control.Label
        Slider_4       matlab.ui.control.Slider
        Label_4        matlab.ui.control.Label
        Button_9       matlab.ui.control.StateButton
        Button_8       matlab.ui.control.Button
        Slider_3       matlab.ui.control.Slider
        Label_3        matlab.ui.control.Label
        Contrast       matlab.ui.control.Slider
        Label_2        matlab.ui.control.Label
        Panel          matlab.ui.container.Panel
        UIAxes2        matlab.ui.control.UIAxes
        DropDown       matlab.ui.control.DropDown
        DropDownLabel  matlab.ui.control.Label
        Light          matlab.ui.control.Slider
        Label          matlab.ui.control.Label
        Button_7       matlab.ui.control.Button
        Button_6       matlab.ui.control.Button
        Button_5       matlab.ui.control.Button
        Button_3       matlab.ui.control.Button
        Button_2       matlab.ui.control.Button
        UIAxes         matlab.ui.control.UIAxes
    end

    
    properties (Access = private)
        file % Description
        Image
%         tmp_file
      
    end
    
    methods (Access = private)
        
        function results = B_filter(~,Img,tempsize,sigma0,sigma1)
            gauss = fspecial('gauss',2*tempsize+1,sigma0);

            [m,n] = size(Img);
            
            for i = 1+ tempsize : m - tempsize
                for j = 1+ tempsize : n - tempsize
                   % 提取处理区域得到梯度敏感矩阵
                   % Img(i - tempsize:i + tempsize,j - tempsize:j + tempsize)
                   % 为卷积区域，Img(i,j))为卷积中心点
                   
                   % 得到灰度差值矩阵，并用高斯函数处理为灰度差越大则最终数值越小的权重矩阵
                   temp = abs(Img(i - tempsize:i + tempsize,j - tempsize:j + tempsize) - Img(i,j));
                   temp = exp(-temp.^2/(2*sigma1^2));
                   
                   %将权重矩阵与高斯滤波器相乘，得到双边滤波器，并将权值和化为一
                   filter = gauss.*temp;
                   filter = filter/sum(filter(:));
                   % 卷积求和
                   Img(i,j) = sum(sum((Img(i - tempsize:i + tempsize,j - tempsize:j + tempsize).*filter)));
                end
            end
                   
            results = Img;

        end
        
        function results = run(app)
             
            light = app.Light.Value;
            contrast = app.Contrast.Value;
                        

            I = imread(app.file);
            
            I = app.SkinWhitening(I);

            light = double(light);
            I = I+light;

            if(app.Button_13.Value == true)
                imwrite(I , 'C:/Users/mibbp/Pictures/Lipstick.jpg');
                app.LipStick();
                I = imread('C:/Users/mibbp/Pictures/Lipstickoutput.jpg');
            end
            

            if(app.Button_9.Value==false)
                I = app.ave(I);
            else
                I = app.BF(I);
            end

            
            
            if(app.Button_11.Value==true)
                imwrite(I,'C:/Users/mibbp/Pictures/pysltest.jpg');
                app.Liquefaction();
                I = imread('C:/Users/mibbp/Pictures/thin.jpg');
            end

            if(app.Button_12.Value==true)
                imwrite(I,'C:/Users/mibbp/Pictures/bytest.jpg');
                app.Mibbp_BigEye();
                I = imread('C:/Users/mibbp/Pictures/bigEye.jpg');
            end


            
            img_duibidu = im2uint16(I);
            img_contrast = immultiply(img_duibidu,contrast);

           
            imshow(img_contrast,'Parent',app.UIAxes2);
            msgbox('完成');
        end
        
        function results = BF(app,I)
            tempsize = round(app.Slider_4.Value); 	%控制高斯滤波器大小的参数

            sigma1 = round(app.Slider_5.Value);   	%控制高斯滤波器的标准差
            
            sigma2 = app.Slider_6.Value;	%控制灰度的敏感性，越大的灰度差，权重越小
            
            %模板补零，便于卷积操作，不然会使得图片区域出现黑边
            img = double(padarray(I,[tempsize,tempsize],0))/255;

            imgr = img(:,:,1);
            imgg = img(:,:,2);
            imgb = img(:,:,3);
            
            img(:,:,1) = app.B_filter(imgr,tempsize,sigma1,sigma2);
            img(:,:,2) = app.B_filter(imgg,tempsize,sigma1,sigma2);
            img(:,:,3) = app.B_filter(imgb,tempsize,sigma1,sigma2);

            results = img;
        end
        
        function results = ABF(app,I)
            rho_smooth = round(app.Slider_11.Value);    % Spatial kernel parameter for smoothing step
            rho_sharp = round(app.Slider_12.Value);     % Spatial kernel parameter for sharpening step
            
            f = I;
            f = double(f);
            
            addpath('./Fast-Adaptive-Bilateral-Filtering/fastABF/');
            
            % Set pixelwise sigma (range kernel parameters) for smoothing
            pixelwise = round(app.Slider_9.Value);
            if(mod(pixelwise,2)==0) 
                pixelwise=pixelwise-1;
            end
            M = mrtv(f,pixelwise);
            sigma_smooth = linearMap(1-M,[0,1],[30,70]);
            sigma_smooth = imdilate(sigma_smooth,strel('disk',2,4));  % Clean up the fine noise
            
            % Apply fast algorithm to smooth textures
            g = f;
            tic;
            for it = 1:2
                out = nan(size(f));
                for k = 1:size(f,3)
                    out(:,:,k) = fastABF(g(:,:,k),rho_smooth,sigma_smooth,[],4);
                end
                g = out;
                sigma_smooth = sigma_smooth*0.8;
            end
            
            % Apply fast algorithm to sharpen edges
            % Large variation in sigma is not required for this step
            g_gray = double(rgb2gray(uint8(g)));
            [zeta,sigma_sharp] = logClassifier(g_gray,rho_sharp,[30,31]);
            for it = 1:2     % Run more iterations for greater sharpening
                for k = 1:size(f,3)
                    g(:,:,k) = fastABF(g(:,:,k),rho_sharp,sigma_sharp,g(:,:,k)+zeta,5);
                end
            end
            toc;


            results = uint8(g);
        end
        
        function results = ave(app,I)
            value = app.Slider_3.Value;
            value = round(value);
            if(value~=0)
                R = I(:,:,1);
                G = I(:,:,2);
                B = I(:,:,3);
                R = medfilt2(R,[3,3]);
                G = medfilt2(G,[3,3]);
                B = medfilt2(B,[3,3]);
    
                R = filter2(fspecial('average',value),R)/255;
                G = filter2(fspecial('average',value),G)/255;
                B = filter2(fspecial('average',value),B)/255;
                I = cat(3,R,G,B);
                results = I;
            else
                results = I;
            end
        end
        
        function results = Liquefaction(app)
            LStrength = round(app.Slider_18.Value); 
            RStrength = round(app.Slider_21.Value); 
            Lcen = round(app.Slider_14.Value);
            Rcen = round(app.Slider_19.Value);
            Lrad = round(app.Slider_16.Value);
            Rrad = round(app.Slider_20.Value);
            Center = round(app.Slider_13.Value);
            py.pytest.main(LStrength, RStrength, Lcen, Rcen, Lrad, Rrad, Center);
        end
        
        function results = Mibbp_BigEye(app)
            LStrength = round(app.Slider_25.Value); 
            RStrength = round(app.Slider_28.Value);
            py.bigEye.main(LStrength, RStrength);
        end
        
        function results = SkinWhitening(app,Img)
            value = app.Slider_29.Value;
            im = Img;
            im1=rgb2ycbcr(im);%将图片的RGB值转换成YCbCr值%
            YY=im1(:,:,1);
            Cb=im1(:,:,2);
            Cr=im1(:,:,3);
            [x, y, z]=size(im);
            tst=zeros(x,y);
            Mb=mean(mean(Cb));
            Mr=mean(mean(Cr));
            %计算Cb、Cr的均方差%
            Tb = Cb-Mb;
            Tr = Cr-Mr;
            Db=sum(sum((Tb).*(Tb)))/(x*y);
            Dr=sum(sum((Tr).*(Tr)))/(x*y);
            %根据阀值的要求提取出near-white区域的像素点%
            cnt=1;    
            for i=1:x
                for j=1:y
                    b1=Cb(i,j)-(Mb+Db*sign(Mb));
                    b2=Cr(i,j)-(1.5*Mr+Dr*sign(Mr));
                    if (b1<abs(1.5*Db) && b2<abs(1.5*Dr))
                       Ciny(cnt)=YY(i,j);
                       tst(i,j)=YY(i,j);
                       cnt=cnt+1;
                    end
                end
            end
            cnt=cnt-1;
            iy=sort(Ciny,'descend');%将提取出的像素点从亮度值大的点到小的点依次排列%
            nn=round(cnt/10);
            Ciny2(1:nn)=iy(1:nn);%提取出near-white区域中10%的亮度值较大的像素点做参考白点%
            %提取出参考白点的RGB三信道的值% 
            mn=min(Ciny2);
            for i=1:x
                for j=1:y
                    if tst(i,j)<mn
                       tst(i,j)=0;
                    else
                       tst(i,j)=1;
                    end
                end
            end
            R=im(:,:,1);
            G=im(:,:,2);
            B=im(:,:,3);
            
            R=double(R).*tst;
            G=double(G).*tst;
            B=double(B).*tst;
            
            %计算参考白点的RGB的均值%
            Rav=mean(mean(R));
            Gav=mean(mean(G));
            Bav=mean(mean(B));
            
            Ymax=double(max(max(YY)))*0.15;%计算出图片的亮度的最大值%
             
            %计算出RGB三信道的增益% 
            Rgain=Ymax/Rav;
            Ggain=Ymax/Gav;
            Bgain=Ymax/Bav;
            
            %通过增益调整图片的RGB三信道%
            im(:, :, 1)=im(:, :, 1) * Rgain * (value / 5);
            im(:, :, 2)=im(:, :, 2) * Ggain * (value / 5);
            im(:, :, 3)=im(:, :, 3) * Bgain * (value / 5);
            
            results = im;

        end
        
        function results = LipStick(app)
            H = round(app.Slider_30.Value); 
            S = round(app.Slider_31.Value);
            py.Lipstick.main(H, S);
        end
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: Button_3
        function Button_3Pushed(app, event)
            msgbox("这是一个简易版美图秀秀","关于","help"); %msgbox（）创建消息对话框
        end

        % Button pushed function: Button_2
        function Button_2Pushed(app, event)
            choice = questdlg("您要关闭吗？","关闭","YES","No","No"); %questdlg（）创建问题对话框
            switch choice
                case "YES"
                    delete(app);
                    return; 
                case "No"
                    return;
            end
        end

        % Callback function
        function ButtonPushed(app, event)
            [filename,pathname] = uigetfile({'*.jpg';'*.png';'*.tif';'*.*'});
            if isequal(filename,0)||isequal(pathname,0)
              errordlg("没有选择文件","错误");
            else
                app.file = strcat(pathname,filename);
            end
            im = imread(app.file);
            imshow(im,'Parent',app.UIAxes);
            
        end

        % Callback function
        function Button_4Pushed(app, event)
            obj = videoinput('macvideo',1,'YCbCr422_960x540');
             h = preview(obj);
             while ishandle(h)
                app.file = getsnapshot(obj);  
                imshow(app.file,'Parent',app.UIAxes);
                drawnow
             end

        end

        % Button down function: UIAxes
        function UIAxesButtonDown(app, event)
            [FileName,PathName] = uiputfile({'*.jpg','JPEG(*.jpg)';...
                                 '*.bmp','Bitmap(*.bmp)';...
                                 '*.gif','GIF(*.gif)';...
                                 '*.*',  'All Files (*.*)'},...
                                 'Save Picture','Untitled');
            if FileName==0
                return;
            else
                h=getframe(app.UIAxes2);
                imwrite(h.cdata,[PathName,FileName]);
            end

        end

        % Button pushed function: Button_6
        function Button_6Pushed(app, event)
      %利用复制副本方式进行保存
          newPanel = uipanel;
          copyobj(app.Panel.Children,newPanel); %复制图形对象及其后代到newPanel中去
          fig2Save = ancestor(newPanel,'figure');%把整个面板和图形再复制一遍
          filter = {'*.jpg';'*.png';'*.bmp';'*.tif'};
          [filename,pathname] = uiputfile(filter);%打开用于保存文件的对话框
          newFilename = fullfile(pathname,filename);%形成文件的绝对路径
          saveas(fig2Save,newFilename);%用save（）函数将图窗保存为特定文件格式
        end

        % Value changed function: Light
        function LightValueChanged(app, event)
            app.run();
            
        end

        % Button pushed function: Button_7
        function Button_7Pushed(app, event)
           
        end

        % Value changed function: DropDown
        function DropDownValueChanged(app, event)
            value = app.DropDown.Value;
            if value=="打开"
                [filename,pathname] = uigetfile({'*.jpg';'*.png';'*.tif';'*.*'});
                if isequal(filename,0)||isequal(pathname,0)
                  errordlg("没有选择文件","错误");
                else
                    app.file = strcat(pathname,filename);
                    app.Image = imread(app.file);
                    imshow(app.Image,'Parent',app.UIAxes);
                end
            elseif value=="拍照"
                obj = videoinput('winvideo',1,'MJPG_640x480');
                h = preview(obj);
                while ishandle(h)
                frame = getsnapshot(obj);  
                imwrite(frame,"C:/Users/mibbp/Pictures/pic.jpg");
                app.file = "C:/Users/mibbp/Pictures/pic.jpg";
                app.Image = imread(app.file);
                imshow(app.file,'Parent',app.UIAxes);
                drawnow
                end
            end

        end

        % Button pushed function: Button_5
        function Button_5Pushed(app, event)
            cla(app.UIAxes2,'reset'); %清除坐标区函数
        end

        % Value changed function: Contrast
        function ContrastValueChanged(app, event)

            app.run();
            
        end

        % Value changed function: Slider_3
        function Slider_3ValueChanged(app, event)
  
            app.run();

        end

        % Button pushed function: Button_8
        function Button_8Pushed(app, event)
%             app.file = app.tmp_file;
            I = app.Image;
            ave1 = fspecial('average',3);
            K = filter2(ave1,I)/255;
            app.Image = K;
            imshow(K,'Parent',app.UIAxes2);
        end

        % Value changed function: Button_9
        function Button_9ValueChanged(app, event)
            
        end

        % Value changed function: Slider_4
        function Slider_4ValueChanged(app, event)

            app.run();

        end

        % Value changed function: Slider_5
        function Slider_5ValueChanged(app, event)

            app.run();

        end

        % Value changed function: Slider_6
        function Slider_6ValueChanged(app, event)
               
            app.run();
            
        end

        % Callback function
        function Slider_11ValueChanged(app, event)
            
            app.run();

        end

        % Callback function
        function Slider_12ValueChanged(app, event)
            
            app.run();
            
        end

        % Callback function
        function Slider_9ValueChanged(app, event)
            
            app.run();
            
        end

        % Callback function
        function Button_10ValueChanged(app, event)


        end

        % Value changed function: Button_11
        function Button_11ValueChanged(app, event)
            
            
        end

        % Value changed function: Slider_14
        function Slider_14ValueChanged(app, event)
            value = app.Slider_14.Value;
            app.run();
        end

        % Value changed function: Slider_16
        function Slider_16ValueChanged(app, event)
            value = app.Slider_16.Value;
            app.run();
        end

        % Value changed function: Slider_19
        function Slider_19ValueChanged(app, event)
            value = app.Slider_19.Value;
            app.run();
        end

        % Value changed function: Slider_20
        function Slider_20ValueChanged(app, event)
            value = app.Slider_20.Value;
            app.run();
        end

        % Value changed function: Slider_13
        function Slider_13ValueChanged(app, event)
            value = app.Slider_13.Value;
            app.run();
        end

        % Value changed function: Slider_18
        function Slider_18ValueChanged(app, event)
            value = app.Slider_18.Value;
            app.run();
        end

        % Value changed function: Slider_25
        function Slider_25ValueChanged(app, event)
            value = app.Slider_25.Value;
            app.run();
        end

        % Value changed function: Slider_28
        function Slider_28ValueChanged(app, event)
            value = app.Slider_28.Value;
            app.run();
        end

        % Value changed function: Slider_29
        function Slider_29ValueChanged(app, event)
            value = app.Slider_29.Value;
            app.run();
        end

        % Value changed function: Slider_30
        function Slider_30ValueChanged(app, event)
            value = app.Slider_30.Value;
            app.run();
        end

        % Value changed function: Slider_31
        function Slider_31ValueChanged(app, event)
            value = app.Slider_31.Value;
            app.run();
        end

        % Value changed function: Slider_21
        function Slider_21ValueChanged(app, event)
            value = app.Slider_21.Value;
            app.run();
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 1677 1104];
            app.UIFigure.Name = 'MATLAB App';

            % Create UIAxes
            app.UIAxes = uiaxes(app.UIFigure);
            title(app.UIAxes, '原图')
            zlabel(app.UIAxes, 'Z')
            app.UIAxes.ButtonDownFcn = createCallbackFcn(app, @UIAxesButtonDown, true);
            app.UIAxes.Position = [93 347 565 609];

            % Create Button_2
            app.Button_2 = uibutton(app.UIFigure, 'push');
            app.Button_2.ButtonPushedFcn = createCallbackFcn(app, @Button_2Pushed, true);
            app.Button_2.FontSize = 20;
            app.Button_2.Position = [591 1058 100 36];
            app.Button_2.Text = '退出';

            % Create Button_3
            app.Button_3 = uibutton(app.UIFigure, 'push');
            app.Button_3.ButtonPushedFcn = createCallbackFcn(app, @Button_3Pushed, true);
            app.Button_3.FontSize = 20;
            app.Button_3.Position = [707 1058 100 36];
            app.Button_3.Text = '关于';

            % Create Button_5
            app.Button_5 = uibutton(app.UIFigure, 'push');
            app.Button_5.ButtonPushedFcn = createCallbackFcn(app, @Button_5Pushed, true);
            app.Button_5.FontSize = 20;
            app.Button_5.Position = [472 1059 100 36];
            app.Button_5.Text = '重置';

            % Create Button_6
            app.Button_6 = uibutton(app.UIFigure, 'push');
            app.Button_6.ButtonPushedFcn = createCallbackFcn(app, @Button_6Pushed, true);
            app.Button_6.FontSize = 20;
            app.Button_6.Position = [236 1059 100 36];
            app.Button_6.Text = '保存';

            % Create Button_7
            app.Button_7 = uibutton(app.UIFigure, 'push');
            app.Button_7.ButtonPushedFcn = createCallbackFcn(app, @Button_7Pushed, true);
            app.Button_7.FontSize = 20;
            app.Button_7.Position = [354 1059 100 36];
            app.Button_7.Text = '截图';

            % Create Label
            app.Label = uilabel(app.UIFigure);
            app.Label.HorizontalAlignment = 'right';
            app.Label.Position = [1388 976 29 22];
            app.Label.Text = '亮度';

            % Create Light
            app.Light = uislider(app.UIFigure);
            app.Light.Limits = [-100 100];
            app.Light.ValueChangedFcn = createCallbackFcn(app, @LightValueChanged, true);
            app.Light.Position = [1438 985 150 3];

            % Create DropDownLabel
            app.DropDownLabel = uilabel(app.UIFigure);
            app.DropDownLabel.HorizontalAlignment = 'right';
            app.DropDownLabel.Position = [12 1067 66 22];
            app.DropDownLabel.Text = 'Drop Down';

            % Create DropDown
            app.DropDown = uidropdown(app.UIFigure);
            app.DropDown.Items = {'打开', '拍照', 'Option 3', 'Option 4'};
            app.DropDown.ValueChangedFcn = createCallbackFcn(app, @DropDownValueChanged, true);
            app.DropDown.FontSize = 20;
            app.DropDown.Position = [93 1063 116 26];
            app.DropDown.Value = '打开';

            % Create Panel
            app.Panel = uipanel(app.UIFigure);
            app.Panel.Title = 'Panel';
            app.Panel.Position = [669 306 684 692];

            % Create UIAxes2
            app.UIAxes2 = uiaxes(app.Panel);
            title(app.UIAxes2, 'Title')
            xlabel(app.UIAxes2, 'X')
            ylabel(app.UIAxes2, 'Y')
            zlabel(app.UIAxes2, 'Z')
            app.UIAxes2.Position = [18 35 604 614];

            % Create Label_2
            app.Label_2 = uilabel(app.UIFigure);
            app.Label_2.HorizontalAlignment = 'right';
            app.Label_2.Position = [1375 920 41 22];
            app.Label_2.Text = '对比度';

            % Create Contrast
            app.Contrast = uislider(app.UIFigure);
            app.Contrast.Limits = [0 3];
            app.Contrast.ValueChangedFcn = createCallbackFcn(app, @ContrastValueChanged, true);
            app.Contrast.Position = [1437 929 150 3];
            app.Contrast.Value = 1;

            % Create Label_3
            app.Label_3 = uilabel(app.UIFigure);
            app.Label_3.HorizontalAlignment = 'right';
            app.Label_3.Position = [1388 861 29 22];
            app.Label_3.Text = '磨皮';

            % Create Slider_3
            app.Slider_3 = uislider(app.UIFigure);
            app.Slider_3.Limits = [0 5];
            app.Slider_3.MajorTicks = [0 1 2 3 4 5];
            app.Slider_3.MajorTickLabels = {'0', '1', '2', '3', '4', '5'};
            app.Slider_3.ValueChangedFcn = createCallbackFcn(app, @Slider_3ValueChanged, true);
            app.Slider_3.Position = [1438 870 150 3];

            % Create Button_8
            app.Button_8 = uibutton(app.UIFigure, 'push');
            app.Button_8.ButtonPushedFcn = createCallbackFcn(app, @Button_8Pushed, true);
            app.Button_8.Position = [1446 51 100 25];
            app.Button_8.Text = '美颜';

            % Create Button_9
            app.Button_9 = uibutton(app.UIFigure, 'state');
            app.Button_9.ValueChangedFcn = createCallbackFcn(app, @Button_9ValueChanged, true);
            app.Button_9.Text = '双边滤波器开关';
            app.Button_9.Position = [1463 511 99 22];

            % Create Label_4
            app.Label_4 = uilabel(app.UIFigure);
            app.Label_4.HorizontalAlignment = 'right';
            app.Label_4.Position = [1363 480 65 22];
            app.Label_4.Text = '卷积核大小';

            % Create Slider_4
            app.Slider_4 = uislider(app.UIFigure);
            app.Slider_4.Limits = [1 5];
            app.Slider_4.MajorTicks = [1 2 3 4 5];
            app.Slider_4.ValueChangedFcn = createCallbackFcn(app, @Slider_4ValueChanged, true);
            app.Slider_4.Position = [1438 489 150 3];
            app.Slider_4.Value = 1;

            % Create Slider_5Label
            app.Slider_5Label = uilabel(app.UIFigure);
            app.Slider_5Label.HorizontalAlignment = 'right';
            app.Slider_5Label.Position = [1375 418 41 22];
            app.Slider_5Label.Text = '标准差';

            % Create Slider_5
            app.Slider_5 = uislider(app.UIFigure);
            app.Slider_5.Limits = [1 5];
            app.Slider_5.MajorTicks = [1 2 3 4 5];
            app.Slider_5.ValueChangedFcn = createCallbackFcn(app, @Slider_5ValueChanged, true);
            app.Slider_5.Position = [1437 427 150 3];
            app.Slider_5.Value = 1;

            % Create Label_5
            app.Label_5 = uilabel(app.UIFigure);
            app.Label_5.HorizontalAlignment = 'right';
            app.Label_5.Position = [1375 354 41 22];
            app.Label_5.Text = '灰敏度';

            % Create Slider_6
            app.Slider_6 = uislider(app.UIFigure);
            app.Slider_6.Limits = [0 1];
            app.Slider_6.ValueChangedFcn = createCallbackFcn(app, @Slider_6ValueChanged, true);
            app.Slider_6.Position = [1437 363 150 3];

            % Create Button_11
            app.Button_11 = uibutton(app.UIFigure, 'state');
            app.Button_11.ValueChangedFcn = createCallbackFcn(app, @Button_11ValueChanged, true);
            app.Button_11.Text = '瘦脸开关';
            app.Button_11.Position = [188 218 100 22];

            % Create Label_12
            app.Label_12 = uilabel(app.UIFigure);
            app.Label_12.HorizontalAlignment = 'right';
            app.Label_12.Position = [365 219 53 22];
            app.Label_12.Text = '液化中心';

            % Create Slider_13
            app.Slider_13 = uislider(app.UIFigure);
            app.Slider_13.Limits = [29 34];
            app.Slider_13.MajorTicks = [29 30 31 32 33 34];
            app.Slider_13.ValueChangedFcn = createCallbackFcn(app, @Slider_13ValueChanged, true);
            app.Slider_13.Position = [439 228 150 3];
            app.Slider_13.Value = 30;

            % Create Label_10
            app.Label_10 = uilabel(app.UIFigure);
            app.Label_10.HorizontalAlignment = 'right';
            app.Label_10.Position = [98 165 53 22];
            app.Label_10.Text = '左脸中心';

            % Create Slider_14
            app.Slider_14 = uislider(app.UIFigure);
            app.Slider_14.Limits = [3 6];
            app.Slider_14.MajorTicks = [3 4 5 6];
            app.Slider_14.MajorTickLabels = {'3', '4', '5', '6'};
            app.Slider_14.ValueChangedFcn = createCallbackFcn(app, @Slider_14ValueChanged, true);
            app.Slider_14.Position = [172 174 150 3];
            app.Slider_14.Value = 4;

            % Create Label_13
            app.Label_13 = uilabel(app.UIFigure);
            app.Label_13.HorizontalAlignment = 'right';
            app.Label_13.Position = [98 108 53 22];
            app.Label_13.Text = '左脸范围';

            % Create Slider_16
            app.Slider_16 = uislider(app.UIFigure);
            app.Slider_16.Limits = [1 5];
            app.Slider_16.MajorTicks = [0 1 2 3 4 5];
            app.Slider_16.MajorTickLabels = {'0', '1', '2', '3', '4', '5'};
            app.Slider_16.ValueChangedFcn = createCallbackFcn(app, @Slider_16ValueChanged, true);
            app.Slider_16.Position = [172 117 150 3];
            app.Slider_16.Value = 2;

            % Create Label_15
            app.Label_15 = uilabel(app.UIFigure);
            app.Label_15.HorizontalAlignment = 'right';
            app.Label_15.Position = [74 42 77 22];
            app.Label_15.Text = '左脸液化强度';

            % Create Slider_18
            app.Slider_18 = uislider(app.UIFigure);
            app.Slider_18.Limits = [10 200];
            app.Slider_18.MajorTicks = [10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200];
            app.Slider_18.MajorTickLabels = {'低', '', '', '', '', '', '', '', '', '中', '', '', '', '', '', '', '', '', '', '强'};
            app.Slider_18.ValueChangedFcn = createCallbackFcn(app, @Slider_18ValueChanged, true);
            app.Slider_18.Position = [172 51 150 3];
            app.Slider_18.Value = 100;

            % Create Label_16
            app.Label_16 = uilabel(app.UIFigure);
            app.Label_16.HorizontalAlignment = 'right';
            app.Label_16.Position = [365 165 53 22];
            app.Label_16.Text = '右脸中心';

            % Create Slider_19
            app.Slider_19 = uislider(app.UIFigure);
            app.Slider_19.Limits = [12 15];
            app.Slider_19.MajorTicks = [12 13 14 15];
            app.Slider_19.MajorTickLabels = {'12', '13', '14', '15'};
            app.Slider_19.ValueChangedFcn = createCallbackFcn(app, @Slider_19ValueChanged, true);
            app.Slider_19.Position = [439 174 150 3];
            app.Slider_19.Value = 13;

            % Create Label_17
            app.Label_17 = uilabel(app.UIFigure);
            app.Label_17.HorizontalAlignment = 'right';
            app.Label_17.Position = [365 108 53 22];
            app.Label_17.Text = '右脸范围';

            % Create Slider_20
            app.Slider_20 = uislider(app.UIFigure);
            app.Slider_20.Limits = [1 5];
            app.Slider_20.MajorTicks = [0 1 2 3 4 5];
            app.Slider_20.MajorTickLabels = {'0', '1', '2', '3', '4', '5'};
            app.Slider_20.ValueChangedFcn = createCallbackFcn(app, @Slider_20ValueChanged, true);
            app.Slider_20.Position = [439 117 150 3];
            app.Slider_20.Value = 2;

            % Create Label_18
            app.Label_18 = uilabel(app.UIFigure);
            app.Label_18.HorizontalAlignment = 'right';
            app.Label_18.Position = [341 42 77 22];
            app.Label_18.Text = '右脸液化强度';

            % Create Slider_21
            app.Slider_21 = uislider(app.UIFigure);
            app.Slider_21.Limits = [10 200];
            app.Slider_21.MajorTicks = [10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200];
            app.Slider_21.MajorTickLabels = {'低', '', '', '', '', '', '', '', '', '中', '', '', '', '', '', '', '', '', '', '强'};
            app.Slider_21.ValueChangedFcn = createCallbackFcn(app, @Slider_21ValueChanged, true);
            app.Slider_21.Position = [439 51 150 3];
            app.Slider_21.Value = 100;

            % Create Button_12
            app.Button_12 = uibutton(app.UIFigure, 'state');
            app.Button_12.Text = '大眼开关';
            app.Button_12.Position = [866 218 100 22];

            % Create Label_19
            app.Label_19 = uilabel(app.UIFigure);
            app.Label_19.HorizontalAlignment = 'right';
            app.Label_19.Position = [643 165 77 22];
            app.Label_19.Text = '左眼放大强度';

            % Create Slider_25
            app.Slider_25 = uislider(app.UIFigure);
            app.Slider_25.MajorTicks = [0 10 20 30 40 50 60 70 80 90 100];
            app.Slider_25.MajorTickLabels = {'低', '', '', '', '', '中', '', '', '', '', '强'};
            app.Slider_25.ValueChangedFcn = createCallbackFcn(app, @Slider_25ValueChanged, true);
            app.Slider_25.Position = [741 174 150 3];
            app.Slider_25.Value = 30;

            % Create Label_20
            app.Label_20 = uilabel(app.UIFigure);
            app.Label_20.HorizontalAlignment = 'right';
            app.Label_20.Position = [927 165 77 22];
            app.Label_20.Text = '右眼放大强度';

            % Create Slider_28
            app.Slider_28 = uislider(app.UIFigure);
            app.Slider_28.MajorTicks = [0 10 20 30 40 50 60 70 80 90 100];
            app.Slider_28.MajorTickLabels = {'低', '', '', '', '', '中', '', '', '', '', '强'};
            app.Slider_28.ValueChangedFcn = createCallbackFcn(app, @Slider_28ValueChanged, true);
            app.Slider_28.Position = [1025 174 150 3];
            app.Slider_28.Value = 30;

            % Create Label_21
            app.Label_21 = uilabel(app.UIFigure);
            app.Label_21.HorizontalAlignment = 'right';
            app.Label_21.Position = [1387 798 29 22];
            app.Label_21.Text = '美白';

            % Create Slider_29
            app.Slider_29 = uislider(app.UIFigure);
            app.Slider_29.Limits = [1 10];
            app.Slider_29.MajorTicks = [1 2 3 4 5 6 7 8 9 10];
            app.Slider_29.MajorTickLabels = {'低', '', '', '', '中', '', '', '', '', '强'};
            app.Slider_29.ValueChangedFcn = createCallbackFcn(app, @Slider_29ValueChanged, true);
            app.Slider_29.Position = [1437 807 150 3];
            app.Slider_29.Value = 5;

            % Create Button_13
            app.Button_13 = uibutton(app.UIFigure, 'state');
            app.Button_13.Text = '唇彩开关';
            app.Button_13.Position = [1463 703 99 22];

            % Create Label_22
            app.Label_22 = uilabel(app.UIFigure);
            app.Label_22.HorizontalAlignment = 'right';
            app.Label_22.Position = [1388 672 29 22];
            app.Label_22.Text = '色相';

            % Create Slider_30
            app.Slider_30 = uislider(app.UIFigure);
            app.Slider_30.Limits = [0 180];
            app.Slider_30.MajorTicks = [0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180];
            app.Slider_30.MajorTickLabels = {'低', '', '', '', '', '', '', '', '', '中', '', '', '', '', '', '', '', '', '强'};
            app.Slider_30.ValueChangedFcn = createCallbackFcn(app, @Slider_30ValueChanged, true);
            app.Slider_30.Position = [1437 681 153 3];
            app.Slider_30.Value = 170;

            % Create Label_23
            app.Label_23 = uilabel(app.UIFigure);
            app.Label_23.HorizontalAlignment = 'right';
            app.Label_23.Position = [1377 610 41 22];
            app.Label_23.Text = '饱和度';

            % Create Slider_31
            app.Slider_31 = uislider(app.UIFigure);
            app.Slider_31.Limits = [150 250];
            app.Slider_31.MajorTicks = [150 160 170 180 190 200 210 220 230 240 250];
            app.Slider_31.MajorTickLabels = {'低', '', '', '', '', '中', '', '', '', '', '强'};
            app.Slider_31.ValueChangedFcn = createCallbackFcn(app, @Slider_31ValueChanged, true);
            app.Slider_31.Position = [1439 619 150 3];
            app.Slider_31.Value = 200;

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = MTXXmlapp

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end