function add_toolboxes()
%ADD_TOOLBOXES Add the project folders and Tensor Toolbox to the path.

matlab_root = fileparts(mfilename("fullpath"));
addpath(genpath(matlab_root));

tensor_toolbox_root = string(getenv("TENSOR_TOOLBOX_ROOT"));
if strlength(tensor_toolbox_root) == 0
    repository_root = fileparts(matlab_root);
    local_candidate = fullfile(repository_root, "tensor_toolbox");
    if isfolder(local_candidate)
        tensor_toolbox_root = string(local_candidate);
    end
end

if strlength(tensor_toolbox_root) == 0 || ~isfolder(tensor_toolbox_root)
    error([ ...
        "Tensor Toolbox was not found. Set TENSOR_TOOLBOX_ROOT to " ...
        "the Tensor Toolbox 3.8 directory before starting MATLAB."]);
end

addpath(char(tensor_toolbox_root));

if exist("ttensor", "class") ~= 8 || exist("tensor", "class") ~= 8
    error("Tensor Toolbox classes were not found after adding the path.");
end

fprintf("Added project MATLAB folders from %s\n", matlab_root);
fprintf("Added Tensor Toolbox from %s\n", tensor_toolbox_root);
end
