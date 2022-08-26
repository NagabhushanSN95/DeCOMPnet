function s_rred = S_RRED_Wrapper(ref_video, dis_video, Nframes, frame_height, frame_width)
% add to path necessary files
addpath(genpath('matlabPyrTools'))

srred = zeros(1, Nframes);
trred = zeros(1, Nframes);

for frame_ind = 1 : Nframes
    if frame_ind < Nframes
        % read i and i+1 frames of reference and distorted video
        ref_frame = read_single_frame(ref_video, frame_ind, frame_height, frame_width);
        ref_frame_next = read_single_frame(ref_video, frame_ind + 1, frame_height, frame_width);
        
        dis_frame = read_single_frame(dis_video, frame_ind, frame_height, frame_width);
        dis_frame_next = read_single_frame(dis_video, frame_ind + 1, frame_height, frame_width);

        [spatial_ref, temporal_ref] = extract_info(ref_frame_next, ref_frame);
        [spatial_dis, temporal_dis] = extract_info(dis_frame_next, dis_frame);

        srred(frame_ind) = mean2(abs(spatial_ref-spatial_dis));
        trred(frame_ind) = mean2(abs(temporal_ref-temporal_dis));
    else
        % cannot read more frame, use previous values
        srred(frame_ind) = srred(frame_ind-1);
        trred(frame_ind) = trred(frame_ind-1);
        
    end;
    
end;

s_rred = mean(srred);
