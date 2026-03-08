function varargout = pluse2x2(varargin)
% PLUSE2X2 MATLAB code for pluse2x2.fig
%      PLUSE2X2, by itself, creates a new PLUSE2X2 or raises the existing
%      singleton*.
%
%      H = PLUSE2X2 returns the handle to a new PLUSE2X2 or the handle to
%      the existing singleton*.
%
%      PLUSE2X2('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PLUSE2X2.M with the given input arguments.
%
%      PLUSE2X2('Property','Value',...) creates a new PLUSE2X2 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before pluse2x2_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to pluse2x2_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help pluse2x2

% Last Modified by GUIDE v2.5 13-Apr-2022 10:12:33

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @pluse2x2_OpeningFcn, ...
                   'gui_OutputFcn',  @pluse2x2_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT
end

% --- Executes just before pluse2x2 is made visible.
function pluse2x2_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to pluse2x2 (see VARARGIN)

% Choose default command line output for pluse2x2
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes pluse2x2 wait for user response (see UIRESUME)
% uiwait(handles.figure1);
end

% --- Outputs from this function are returned to the command line.
function varargout = pluse2x2_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
end

% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global  dev
global  cfg
cfg = initcfg();
cfg.figure1=handles.axes1;
cfg.figure2=handles.axes2;
%cfg.figure3=handles.axes3;

dev = daq.createSession('ni');
dev.DurationInSeconds = cfg.duration;
dev.IsContinuous = 1;

for i=1:cfg.nout
    addAnalogOutputChannel(dev,cfg.dev,cfg.output{i},'Voltage');
end

for i=1:cfg.nin
    ch= addAnalogInputChannel(dev,cfg.dev,cfg.input{i},'Voltage');
    ch.Range=[-5,5];
end

dev.Rate = cfg.fs;
queueOutputData(dev,cfg.dataout);
dev.NotifyWhenDataAvailableExceeds = cfg.notifysample;
dev.NotifyWhenScansQueuedBelow = cfg.outlength/2;

cfg.inputlistener = addlistener(dev,'DataAvailable',@processData);
cfg.outputlistener = addlistener(dev,'DataRequired',@queueMoreData);

startBackground(dev);

fprintf('\nstart test\n');
end

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global  dev
global  cfg
daq.reset;
delete (cfg.inputlistener);
delete(cfg.outputlistener);
save dataout.mat cfg
end

function queueMoreData(src,event)
global  dev
global  cfg
%fprintf('adddata\n');
queueOutputData(dev,cfg.dataout);
end


function processData(src,event)
global  dev
global  cfg
cfg.index = cfg.index + 1;
datalen = size(event.Data,1);

if(mod(cfg.index,10)==0)
    fprintf('index %d, datalength %d \n',cfg.index,datalen)
end
dataseg=event.Data;
%dataseg=dataseg(:,1);

if(cfg.index==1)
    cfg.dc=mean(dataseg(1:(cfg.fs/50),:),1);
else
    cfg.dc=cfg.dc*0.8+mean(dataseg(1:(cfg.fs/50),:),1)*0.2;
end
 
dataseg=dataseg-cfg.dc;  %remove microphone DC
datalen=size(dataseg,1);
cfg.rawdata(1:end-datalen,:)=cfg.rawdata(datalen+1:end,:);
cfg.rawdata(end-datalen+1:end,:)=dataseg;
cfg.samples=cfg.samples+size(dataseg,1);

% 
% cir1=zeros(cfg.nin,cfg.zclen,cfg.zcrep);  %we have two frames 2048 points for two mics
% cir2=zeros(cfg.nin,cfg.zclen,cfg.zcrep); 
% for i=1:cfg.nin                  %for each mic
%     data=reshape(dataseg(:,i),cfg.zclen,cfg.zcrep);  %make the data to be 1024*2
%     data_fft=fft(data,[],1);                 %FFT
%     temp_fft1=zeros(size(data));
%     temp_fft2=zeros(size(data));
%     for j=1:size(temp_fft1,2)                 %correlation on frequency domain with ZC
%         if(cfg.zc_l==1)
%             temp_fft1(1,j)=data_fft(cfg.startpoint,j);
%         else
%              temp_fft1((cfg.zclen/2+1-(cfg.zc_l-1)/2):2:(cfg.zclen/2+1+(cfg.zc_l-1)/2),j)=data_fft((cfg.startpoint(1)-(cfg.zc_l-1)/2):2:(cfg.startpoint(1)+(cfg.zc_l-1)/2),j).*cfg.zc_fft(1:2:end)';
%              temp_fft2((cfg.zclen/2+1-(cfg.zc_l-1)/2):2:(cfg.zclen/2+1+(cfg.zc_l-1)/2),j)=data_fft((cfg.startpoint(2)-(cfg.zc_l-1)/2):2:(cfg.startpoint(2)+(cfg.zc_l-1)/2),j).*cfg.zc_fft(1:2:end)';
%         end
%     end
%     cir1(i,:,:)=ifft(fftshift(temp_fft1,1),[],1);  %ifft and get the CIR
%     cir2(i,:,:)=ifft(fftshift(temp_fft2,1),[],1);  %ifft and get the CIR
% end
% % if(cfg.index==20)
% %     cfg.static=squeeze(cir(:,:,1));
% % end
% % ratio=8;
% 
% if(cfg.index>20)
%     if(cfg.index<cfg.peakseglen/cfg.zcrep)
%         cfg.peaks(1:end-cfg.zcrep)=cfg.peaks(end);
%         peaksdc=mean(cfg.peaks(end-cfg.peakseglen:end));
%         cfg.peakschest(1:end-cfg.zcrep)=cfg.peakschest(end);
%         peakschestdc=mean(cfg.peakschest(end-cfg.peakseglen:end));
%         angcorr=1;
%     else
%         peaksdc=mean(cfg.peaks(end-cfg.peakseglen:end));
%         peakschestdc=mean(cfg.peakschest(end-cfg.peakseglen:end));
%         nodcdiffmax=max(diff(cfg.peaksnodc(end-cfg.peakseglen:end)));
%         nodcmax=max(cfg.peaksnodc(end-cfg.peakseglen:end));
%         if( abs(angle(conj(nodcmax)*nodcdiffmax))>pi/2)
%             nodcmax=-nodcmax;
%         end
%         angcorr=conj(nodcmax)/abs(nodcmax);
%         cfg.peaks(1:end-cfg.zcrep)=cfg.peaks(cfg.zcrep+1:end);
%         cfg.peaksnodc(1:end-cfg.zcrep)=cfg.peaksnodc(cfg.zcrep+1:end);
%         cfg.peakschest(1:end-cfg.zcrep)=cfg.peakschest(cfg.zcrep+1:end);
%         cfg.peakschestnodc(1:end-cfg.zcrep)=cfg.peakschestnodc(cfg.zcrep+1:end);
%     end
%     
%     [~,ind]=max(abs(cir1(1,:,1)));
%     ind=20;
%     if(cfg.zc_l==1)
%         cfg.peaks(end-cfg.zcrep+1:end)=temp_fft1(1,:);
%         cfg.peaksnodc(end-cfg.zcrep+1:end)=temp_fft1(1,:)-peaksdc;
%     else
%         cfg.peaks(end-cfg.zcrep+1:end)=squeeze(mean(cir2(2,ind-5:ind,:)));
%         cfg.peaksnodc(end-cfg.zcrep+1:end)=cfg.peaks(end-cfg.zcrep+1:end)-peaksdc;
%     end
%     chestpos=20;
%     [~,ind]=max(abs(cir1(1,chestpos:end-chestpos,1)));
%     cfg.peakschest(end-cfg.zcrep+1:end)=squeeze(mean(cir1(1,ind+chestpos-5:ind+chestpos+5,:)));
%     cfg.peakschestnodc(end-cfg.zcrep+1:end)=cfg.peakschest(end-cfg.zcrep+1:end)-peakschestdc;
%           %plot(cfg.figure1,squeeze(angle(cir(1,1:cfg.zclen/4,1))));
%     %plot(cfg.figure1,(1:cfg.zclen/ratio)/cfg.fs*cfg.soundspeed/2-cfg.startdis,squeeze(abs(cir(1,1:cfg.zclen/ratio,1)-cfg.static(1,1:cfg.zclen/ratio))));
% %     if(mod(cfg.index,10)~=0)
% %         return;
% %     end
%     
% %     axis(cfg.figure1,[0 250 -pi pi]);
% %     hold(cfg.figure1, 'on');
% %     plot(cfg.figure1,angle(cir(2,1:cfg.zclen/ratio,1)));
%     
%     %text(cfg.figure1,10,2.5,labels{curlabel},'FontSize',48,'Color','r');
%     %plot(cfg.figure1,squeeze(angle(cir(2,1:cfg.zclen/4,1))));
%     %plot(cfg.figure1,(1:cfg.zclen/ratio)/cfg.fs*cfg.soundspeed/2-cfg.startdis,squeeze(abs(cir(2,1:cfg.zclen/ratio,1)-cfg.static(2,1:cfg.zclen/ratio))));
%     %hold(cfg.figure1, 'off');
%     %axis(cfg.figure1,[-cfg.startdis 20 0 0.25]);
%     %axis(cfg.figure1,[10 40 0 0.05]);
%     fl=2;
%     myf=ones(1,fl)/fl;
% %     a=conv(real(cfg.peaks-mean(cfg.peaks)),myf,'same');
% %     b=conv(imag(cfg.peaks-mean(cfg.peaks)),myf,'same');
%     %a=conv(real(cfg.peaksnodc.*angcorr),myf,'same');
%     a=(real(cfg.peaksnodc.*angcorr));
%     b=conv(-real(cfg.peakschestnodc.*angcorr),myf,'same')*5; %elbow take negative?
%     
%     
%     %plot(cfg.figure1,abs(cir(1,1:cfg.zclen/ratio,1)));
%     
%     plot(cfg.figure2,(fl:cfg.peaklen-fl)*cfg.zclen/cfg.fs,a(fl:end-fl));
%      hold(cfg.figure2, 'on');
%      plot(cfg.figure2,(fl:cfg.peaklen-fl)*cfg.zclen/cfg.fs,b(fl:end-fl));
%      hold(cfg.figure2, 'off');
%     %axis(cfg.figure2,[10 40 0 0.05]);
%     %axis(cfg.figure2,[-cfg.startdis 20 0 0.25]);
%     mycorr=abs(xcorr(a));
%     mycorr=abs(mycorr(cfg.peaklen+1:end));
% 
%    % plot(cfg.figure1,(-length(a)+1:length(b)-1),xcorr(a,b));
%    [~,idx]=max(xcorr(a-mean(a),b-mean(b)));
%    delay=(idx-length(a)+1)*cfg.zclen/96;
%     a=abs(fft(a));
%     %plot(cfg.figure1,mycorr(1:100));
%     plot(cfg.figure1,abs(cir1(1,1:400,1)));
%     hold(cfg.figure1,'on');
%     plot(cfg.figure1,abs(cir2(2,1:400,1)));
%     plot(cfg.figure1,abs(cir1(2,1:400,1)));
%     plot(cfg.figure1,abs(cir2(1,1:400,1)));
%     hold(cfg.figure1,'off');
%     %plot(cfg.figure1,((1:100)-1)/cfg.peaklen/cfg.zclen*cfg.fs,a(1:100));
%     
%     peakstart=cfg.peakrate/2;
%     [pks,loc]=findpeaks(mycorr(peakstart+1:end));
%     [~,idx]=max(pks);
%     %fprintf("peal=%d\n",loc(1));
%     if(pks(idx)>0.5*mycorr(1))
%         if( (loc(idx)+peakstart)<cfg.peakrate*1.5)
%             heartrate= cfg.peakrate/(loc(idx)+peakstart)*60;
%             %fprintf("heart rate=%d\n",heartrate);
%             yl=ylim(cfg.figure1);
%             text(cfg.figure1,0,0.8*yl(2),['Heart rate ' num2str(heartrate),' bpm'],'FontSize',30,'Color','r');
%             text(cfg.figure1,0,0.5*yl(2),['Delay ' num2str(delay),' ms'],'FontSize',30,'Color','r');
%         end
%     end
%     
%     drawnow();
% end
end
