function [values, info] = apply_tucker_row_khatri_rao_sketch(X, sketch)
%APPLY_TUCKER_ROW_KHATRI_RAO_SKETCH Contract a Tucker tensor with S.
%
% The contraction uses only the Tucker core, factor matrices, and stored
% sketch factors. It does not form vec(X) or the full sketching matrix.

callTimer = tic;

if ~isa(X, 'ttensor')
    error('X must be a Tensor Toolbox ttensor object.');
end

requiredFields = {'tensor_dimensions', 'number_of_modes', ...
    'sketch_size', 'factors'};

for fieldIndex = 1:numel(requiredFields)
    if ~isfield(sketch, requiredFields{fieldIndex})
        error('The residual sketch structure is incomplete.');
    end
end

if ~isequal(double(size(X)), double(sketch.tensor_dimensions))
    error('The Tucker tensor dimensions do not match the sketch.');
end

numberOfModes = sketch.number_of_modes;
sketchSize = sketch.sketch_size;
projectedFactors = cell(numberOfModes, 1);

componentTimer = tic;
for mode = 1:numberOfModes
    projectedFactors{mode} = sketch.factors{mode} * X.u{mode};
end
factorProjectionTime = toc(componentTimer);

componentTimer = tic;
values = zeros(sketchSize, 1);
vectors = cell(numberOfModes, 1);

for rowIndex = 1:sketchSize
    for mode = 1:numberOfModes
        vectors{mode} = projectedFactors{mode}(rowIndex, :).';
    end
    values(rowIndex) = ttv(X.core, vectors);
end
coreContractionTime = toc(componentTimer);

info.factor_projection_time_sec = factorProjectionTime;
info.core_contraction_time_sec = coreContractionTime;
info.call_time_sec = toc(callTimer);

end
