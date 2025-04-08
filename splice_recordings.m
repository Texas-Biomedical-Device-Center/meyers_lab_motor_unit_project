% Define input and output directories
input_folder = '/Users/ecm081000/Library/CloudStorage/Box-Box/Python/Projects/swarm-contrastive-decomposition/data/input/';
output_folder = '/Users/ecm081000/Library/CloudStorage/Box-Box/Python/Projects/swarm-contrastive-decomposition/data/input_chunked';

% Get a list of all .mat files in the input folder
mat_files = dir(fullfile(input_folder, '*.mat'));

% Loop through each .mat file in the folder
for file_idx = 1:length(mat_files)
    
    % Load the current .mat file
    file_path = fullfile(input_folder, mat_files(file_idx).name);
    data = load(file_path);
    
    % Assume the loaded struct contains the 'signal' field
    signal = data.signal;
    
    % Extract important values
    fsamp = 4000;                  % Sampling frequency (4000 Hz)
    stim_time = signal.stim_time;  % Stimulation times (in seconds)
    chunk_size = 10 * fsamp;       % Desired chunk size (10 seconds)
    min_chunk_size = 5 * fsamp;    % Minimum chunk size (5 seconds)
    exclude_time = 1;              % 1-second exclusion window
    exclude_samples = exclude_time * fsamp;

    % Signal data and dimensions
    signal_data = signal.data;             
    [num_channels, total_samples] = size(signal_data);

    % Initialize chunk counter
    chunk_idx = 1;
    current_idx = 1;  % Start of the first chunk
    
    % Store chunk data to allow for merging if needed
    chunk_list = {};

    % Convert stim times to sample indices
    stim_samples = round(stim_time * fsamp);

    % Loop through the data in 10-second blocks
    while current_idx <= total_samples

        % Define the end index for the current chunk (initially set to 10s)
        end_idx = min(current_idx + chunk_size - 1, total_samples);

        % Check for stim proximity near the end of the chunk
        proximity_check = abs(stim_samples - end_idx) < exclude_samples;

        % If too close, extend the chunk until it clears the 1-second window
        while any(proximity_check) && (end_idx - current_idx + 1) < 2 * chunk_size
            % Extend the end point to avoid stim proximity
            end_idx = end_idx + fsamp;  % Extend by 1 second
            % Prevent exceeding the total sample length
            if end_idx >= total_samples
                end_idx = total_samples;
                break;
            end
            % Recheck for proximity
            proximity_check = abs(stim_samples - end_idx) < exclude_samples;
        end

        % Extract the chunk
        chunk_data = signal_data(:, current_idx:end_idx);

        % Store the chunk for merging check
        chunk_list{chunk_idx} = chunk_data; %#ok<AGROW>

        % Increment the chunk counter
        chunk_idx = chunk_idx + 1;
        
        % Move to the next chunk
        current_idx = end_idx + 1;

    end

    % Post-processing: Merge small final chunk with the previous one if needed
    if length(chunk_list) > 1
        final_chunk_length = size(chunk_list{end}, 2);
        
        % If final chunk is less than 5s, merge it with the previous chunk
        if final_chunk_length < min_chunk_size
            % Merge the last two chunks
            chunk_list{end-1} = [chunk_list{end-1}, chunk_list{end}];
            % Remove the small final chunk
            chunk_list(end) = [];
        end
    end

    % Save the processed chunks to files
    for i = 1:length(chunk_list)
        chunk_signal = signal;
        chunk_signal.data = chunk_list{i};
        
        % Generate a unique filename for each chunk
        [~, name, ~] = fileparts(mat_files(file_idx).name);
        filename = sprintf('%s_%d.mat', name, i);
        
        % Save the chunk to the output folder
        save(fullfile(output_folder, filename), 'chunk_signal');
    end
    
    fprintf('Finished processing file: %s\n', mat_files(file_idx).name);
end

disp('All files processed and chunks saved!');