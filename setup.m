function repositoryRoot = setup(tensorToolboxRoot)
%SETUP Add the dissertation code and Tensor Toolbox to the MATLAB path.
%
%   setup() adds every classified source folder in this repository. If the
%   environment variable TENSOR_TOOLBOX_ROOT is set, that folder is added
%   too. A Tensor Toolbox installation already on the MATLAB path is also
%   accepted.
%
%   setup(tensorToolboxRoot) explicitly supplies the Tensor Toolbox folder.

repositoryRoot = fileparts(mfilename('fullpath'));
addpath(genpath(repositoryRoot));

if nargin < 1 || strlength(string(tensorToolboxRoot)) == 0
    tensorToolboxRoot = getenv('TENSOR_TOOLBOX_ROOT');
end

if strlength(string(tensorToolboxRoot)) > 0
    tensorToolboxRoot = char(tensorToolboxRoot);
    if ~isfolder(tensorToolboxRoot)
        error('Tensor Toolbox folder not found: %s', tensorToolboxRoot);
    end
    addpath(tensorToolboxRoot);
end

if isempty(which('tensor')) || isempty(which('ttensor'))
    warning('low-rank-tensor-pde:TensorToolboxMissing', ...
        ['Tensor Toolbox is not on the MATLAB path. Matrix and vector ', ...
        'routines remain available, but tensor algorithms require it. ', ...
        'Call setup(''/path/to/tensor_toolbox'') or set ', ...
        'TENSOR_TOOLBOX_ROOT.']);
else
    fprintf('Tensor Toolbox found: %s\n', which('tensor'));
end

fprintf('Dissertation code added: %s\n', repositoryRoot);

end
