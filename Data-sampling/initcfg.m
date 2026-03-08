function niconfig = initcfg()

niconfig.fs = 192000;  %sampling frequency
niconfig.volume =10;  %output Vpp/2  %1

niconfig.dev = 'Dev1';
niconfig.nout=2;
niconfig.output={'ao0','ao1'};  %NI output ports

niconfig.nin=4;
niconfig.input={'ai0','ai1','ai2','ai3'}; %NI input ports

niconfig.figure1=[];
niconfig.figure2=[];
niconfig.figure3=[];

niconfig.inputlistener=[];
niconfig.outputlistener=[];

%%%　ｍａｉｎ
niconfig.outlength = niconfig.fs; %output sample buffer, one second
niconfig.freq =39000;  %27000 %central frequency %39000
niconfig.zclen=1920/2;   %FFT size 
niconfig.zc_l=102;  %10kHz bandwidth, must be even
niconfig.zc_u=[7,13];       %ZC u
niconfig.zcrep = 4*1920/niconfig.zclen;


niconfig.seglen = niconfig.zclen*niconfig.zcrep;
niconfig.notifysample = niconfig.seglen;
niconfig.notifytime = niconfig.notifysample/niconfig.fs;
wind=hamming(niconfig.zc_l/2);  
niconfig.duration = 120; %Total testing time in seconds
for i=1:length(niconfig.zc_u)
    if(niconfig.zc_l>1)
        zcseq(:,i) = zadoff_chu(niconfig.zc_l/2,niconfig.zc_u(i)); %generate ZC in time
        niconfig.zc_fft(:,i)=fftshift(fft(zcseq(:,i))).*wind; 
    else
        niconfig.zc_fft=1;
    end
end
ofdm=zeros(niconfig.zclen,2);                    %OFDM symbol in spectrum, 1024 points

%% 找到奇偶对应频点
niconfig.startpoint(1)=floor(niconfig.freq/niconfig.fs*niconfig.zclen);  %find the center subcarrier for the given central freq
niconfig.total_point=floor((niconfig.startpoint(1)-(niconfig.zc_l-1)/2:niconfig.startpoint(1)+(niconfig.zc_l-1)/2));
niconfig.odd_point=(niconfig.total_point(1:2:end));
niconfig.even_point=(niconfig.total_point(2:2:end));
ofdm(niconfig.odd_point,1)=niconfig.zc_fft(:,1);
ofdm(niconfig.even_point,2)=niconfig.zc_fft(:,2);



niconfig.zcseq=zeros(niconfig.zclen,2);
ofdm(end:-1:(niconfig.zclen/2+2),:)=conj(ofdm(2:niconfig.zclen/2,:));
niconfig.zcseq(:,1)=ifft(ofdm(:,1));  % OFDM symbol in time
niconfig.zcseq(:,2)=ifft(ofdm(:,2));



 
niconfig.dataout =repmat(real(niconfig.zcseq),100,1);           % repeat 100 symbols longer than one second
niconfig.dataout=niconfig.dataout./max(abs(niconfig.dataout))*niconfig.volume;  % adjust Vpp


niconfig.index = 0;                             %record for frames
niconfig.samples = 1;                           %record for samples
niconfig.dc=ones(1,niconfig.nin)*0.78;          %microphone DC offset
niconfig.rawdata =zeros(niconfig.fs*15,niconfig.nin);

niconfig.temp=20;
niconfig.soundspeed=(331.3+0.606*niconfig.temp)*100;
niconfig.wavelength= niconfig.soundspeed/niconfig.freq;  %temperature and wavelength

niconfig.static=zeros(niconfig.zclen,niconfig.nin);
niconfig.peaklen=500*1920/niconfig.zclen;
niconfig.peaks=zeros(1,niconfig.peaklen);
niconfig.peaksnodc=zeros(1,niconfig.peaklen);
niconfig.peakschest=zeros(1,niconfig.peaklen);
niconfig.peakschestnodc=zeros(1,niconfig.peaklen);
niconfig.peakseglen=2*niconfig.fs/niconfig.zclen;
niconfig.peakrate=niconfig.fs/niconfig.zclen;
end




