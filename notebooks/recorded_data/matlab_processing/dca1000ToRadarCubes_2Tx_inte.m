function radarCubes = dca1000ToRadarCubes_2Tx_inte(rawdata, n_channels, samples_per_chirp, chirps_per_frame,n_frames)
% n_channels = 12;
% samples_per_chirp = 96;
% chirps_per_frame = 128;
% n = size(rawdata,2);
% n_frames = n/(samples_per_chirp*chirps_per_frame*2);
%% 
Rx1 = rawdata(1,:);
Rx2 = rawdata(2,:);
Rx3 = rawdata(3,:);
Rx4 = rawdata(4,:);
Up = size(rawdata,2);
ReadCube = linspace(1,Up,Up);
ReadCube = reshape(ReadCube, [samples_per_chirp,chirps_per_frame*n_frames*2]);
ReadCube = ReadCube';
%%samples_per_chirp
ch1 = [];
ch2 = [];
ch3 = [];
ch4 = [];
ch5 = [];
ch6 = [];
ch7 = [];
ch8 = [];
ch9 = [];
ch10 = [];
ch11 = [];
ch12 = [];
for i = 1:2:chirps_per_frame*n_frames*2
    j=i;
    ch1 = [ch1,Rx1(ReadCube(j,:))];
    ch4 = [ch4,Rx2(ReadCube(j,:))];
    ch7 = [ch7,Rx3(ReadCube(j,:))];
    ch10 = [ch10,Rx4(ReadCube(j,:))];
    j=i+1;
    ch2 = [ch2,Rx1(ReadCube(j,:))];
    ch5 = [ch5,Rx2(ReadCube(j,:))];
    ch8 = [ch8,Rx3(ReadCube(j,:))];
    ch11 = [ch11,Rx4(ReadCube(j,:))];
%     j=i+2;
%     ch3 = [ch3,Rx1(ReadCube(j,:))];
%     ch6 = [ch6,Rx2(ReadCube(j,:))];
%     ch9 = [ch9,Rx3(ReadCube(j,:))];
%     ch12 = [ch12,Rx4(ReadCube(j,:))];
end
%  A=zeros(1, samples_per_chirp, N_ad,n_frames);
% ch = [ch1;ch2;ch3;ch4;ch5;ch6;ch7;ch8;ch9;ch10;ch11;ch12];
%%
ch1x = reshape(ch1, [1, samples_per_chirp, chirps_per_frame, n_frames]);

ch2x = reshape(ch2, [1, samples_per_chirp, chirps_per_frame,n_frames]);
% ch3x = reshape(ch3, [1, samples_per_chirp, chirps_per_frame* n_frames]);
ch4x = reshape(ch4, [1, samples_per_chirp, chirps_per_frame, n_frames]);
ch5x = reshape(ch5, [1, samples_per_chirp, chirps_per_frame, n_frames]);
% ch6x = reshape(ch6, [1, samples_per_chirp, chirps_per_frame*n_frames]);
ch7x = reshape(ch7, [1, samples_per_chirp, chirps_per_frame, n_frames]);
ch8x = reshape(ch8, [1, samples_per_chirp, chirps_per_frame,n_frames]);
% ch9x = reshape(ch9, [1, samples_per_chirp, chirps_per_frame*n_frames]);
ch10x = reshape(ch10, [1, samples_per_chirp, chirps_per_frame, n_frames]);
ch11x = reshape(ch11, [1, samples_per_chirp, chirps_per_frame,n_frames]);
% ch12x = reshape(ch12, [1, samples_per_chirp, chirps_per_frame* n_frames]);
% ch1x=cat(3,ch1x,A);
% ch2x=cat(3,ch2x,A);
% ch4x=cat(3,ch4x,A);
% ch5x=cat(3,ch5x,A);
% ch7x=cat(3,ch7x,A);
% ch8x=cat(3,ch8x,A);
% ch10x=cat(3,ch10x,A);
% ch11x=cat(3,ch11x,A);
%%
% radarCubes = [ch1x;ch2x;ch3x;ch4x;ch5x;ch6x;ch7x;ch8x;ch9x;ch10x;ch11x;ch12x];
radarCubes = [ch1x;ch2x;ch4x;ch5x;ch7x;ch8x;ch10x;ch11x];

end