function add_toolboxes()
%ADD_TOOLBOXES Add the project folders and Tensor Toolbox to the path.

matlabRoot = fileparts(mfilename("fullpath"));
addpath(genpath(matlabRoot));

tensorToolboxRoot = string(getenv("TENSOR_TOOLBOX_ROOT"));
if strlength(tensorToolboxRoot) == 0
    repositoryRoot = fileparts(matlabRoot);
    localCandidate = fullfile(repositoryRoot, "tensor_toolbox");
    if isfolder(localCandidate)
        tensorToolboxRoot = string(localCandidate);
    end
end

if strlength(tensorToolboxRoot) == 0 || ~isfolder(tensorToolboxRoot)
    error([ ...
        "Tensor Toolbox was not found. Set TENSOR_TOOLBOX_ROOT to " ...
        "the Tensor Toolbox 3.8 directory before starting MATLAB."]);
end

addpath(char(tensorToolboxRoot));

if exist("ttensor", "class") ~= 8 || exist("tensor", "class") ~= 8
    error("Tensor Toolbox classes were not found after adding the path.");
end

fprintf("Added project MATLAB folders from %s\n", matlabRoot);
fprintf("Added Tensor Toolbox from %s\n", tensorToolboxRoot);
end
