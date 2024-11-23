outputFolder = 'Diego'; 
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

duration = 1;
fs = 44100;

names = ["uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez"];

recorder = audiorecorder(fs, 16, 1);

for i = 1:10
    fprintf('Preparando para grabar audio %d: "%s"\n', i, names(i));
    
    for count = 1:3
        fprintf('%d\n', count);
        pause(1); 
    end
    
    fprintf('Grabando audio %d...\n', i);
    recordblocking(recorder, duration);
    audioData = getaudiodata(recorder);

    outputFileName = fullfile(outputFolder, sprintf('%d_%s.wav', i, outputFolder));
    audiowrite(outputFileName, audioData, fs);
    
    fprintf('Audio %d guardado como "%d_%s.wav"\n', i, i, outputFolder);
end

fprintf('Todos los audios han sido grabados y guardados en la carpeta "%s".\n', outputFolder);
