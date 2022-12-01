NSUBJECTS=100;
cwd=pwd;
FUNCTIONAL_FILE=cellstr(conn_dir('sub-*_func_sub-*_rest.nii.gz'));
STRUCTURAL_FILE=cellstr(conn_dir('sub-*_anat_sub-*mprage.nii.gz'));
%if rem(length(FUNCTIONAL_FILE),NSUBJECTS),error('mismatch number of functional files %n', length(FUNCTIONAL_FILE));end
%if rem(length(STRUCTURAL_FILE),NSUBJECTS),error('mismatch number of anatomical files %n', length(FUNCTIONAL_FILE));end
nsessions=length(FUNCTIONAL_FILE)/NSUBJECTS;
FUNCTIONAL_FILE=reshape(FUNCTIONAL_FILE,[NSUBJECTS,nsessions]);
STRUCTURAL_FILE={STRUCTURAL_FILE{1:NSUBJECTS}};
disp([num2str(size(FUNCTIONAL_FILE,1)),' subjects']);
disp([num2str(size(FUNCTIONAL_FILE,2)),' sessions']);
TR=2; % Repetition time

% CONN Setup.preprocessing                               (realignment/coregistration/segmentation/normalization/smoothing)
batch.Setup.isnew=1;
batch.Setup.nsubjects=NSUBJECTS;
batch.Setup.RT=TR;                                        % TR (seconds)
batch.Setup.functionals=repmat({{}},[NSUBJECTS,1]);       % Gives copy in array for functional volumnes
for nsub=1:NSUBJECTS,for nses=1:nsessions,batch.Setup.functionals{nsub}{nses}{1}=FUNCTIONAL_FILE{nsub,nses}; end; end %note: each subject's data is defined by three sessions and one single (4d) file per session
batch.Setup.structurals=STRUCTURAL_FILE;                  % Point to anatomical volumes for each subject
nconditions=nsessions;                                  % treats each session as a different condition (comment the following three lines and lines 84-86 below if you do not wish to analyze between-session differences)
if nconditions==1
    batch.Setup.conditions.names={'rest'};
    for ncond=1,for nsub=1:NSUBJECTS,for nses=1:nsessions, batch.Setup.conditions.onsets{ncond}{nsub}{nses}=0; batch.Setup.conditions.durations{ncond}{nsub}{nses}=inf;end;end;end     % rest condition (all sessions)
else
    batch.Setup.conditions.names=[{'rest'}, arrayfun(@(n)sprintf('Session%d',n),1:nconditions,'uni',0)]; % applies the function func to the elements of A
    for ncond=1,for nsub=1:NSUBJECTS,for nses=1:nsessions,              batch.Setup.conditions.onsets{ncond}{nsub}{nses}=0; batch.Setup.conditions.durations{ncond}{nsub}{nses}=inf;end;end;end     % rest condition (all sessions)
    for ncond=1:nconditions,for nsub=1:NSUBJECTS,for nses=1:nsessions,  batch.Setup.conditions.onsets{1+ncond}{nsub}{nses}=[];batch.Setup.conditions.durations{1+ncond}{nsub}{nses}=[]; end;end;end
    for ncond=1:nconditions,for nsub=1:NSUBJECTS,for nses=ncond,        batch.Setup.conditions.onsets{1+ncond}{nsub}{nses}=0; batch.Setup.conditions.durations{1+ncond}{nsub}{nses}=inf;end;end;end % session-specific conditions
end
batch.Setup.preprocessing.steps='default_mni';
batch.Setup.preprocessing.sliceorder='interleaved (Siemens)';
batch.Setup.done=1;
batch.Setup.overwrite='Yes';

% CONN Denoising                                    % Default options (uses White Matter+CSF+realignment+scrubbing+conditions as confound regressors); see conn_batch for additional options
batch.Denoising.filter=[0.008, 0.09];                 % frequency filter (band-pass values, in Hz)
batch.Denoising.done=1;
batch.Denoising.overwrite='Yes';