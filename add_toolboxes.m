function repositoryRoot = add_toolboxes(tensorToolboxRoot)
%ADD_TOOLBOXES Compatibility setup used by experiments and tests.

if nargin < 1
    repositoryRoot = setup();
else
    repositoryRoot = setup(tensorToolboxRoot);
end

if isempty(which('tensor')) || isempty(which('ttensor'))
    error(['Tensor Toolbox is required for this experiment or test. ', ...
        'Set TENSOR_TOOLBOX_ROOT or pass its folder to setup.']);
end

end
