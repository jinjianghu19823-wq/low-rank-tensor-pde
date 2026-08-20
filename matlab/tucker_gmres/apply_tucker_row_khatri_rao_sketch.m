function [values, info] = apply_tucker_row_khatri_rao_sketch(X, sketch)
%APPLY_TUCKER_ROW_KHATRI_RAO_SKETCH Contract a Tucker tensor with S.

call_timer = tic;

if ~isa(X, 'ttensor')
    error('X must be a Tensor Toolbox ttensor object.');
end

required_fields = {'tensor_dimensions', 'number_of_modes', ...
    'sketch_size', 'factors'};

for field_idx = 1:numel(required_fields)
    if ~isfield(sketch, required_fields{field_idx})
        error('The residual sketch structure is incomplete.');
    end
end

if ~isequal(double(size(X)), double(sketch.tensor_dimensions))
    error('The Tucker tensor dimensions do not match the sketch.');
end

d = sketch.number_of_modes;
sketch_size = sketch.sketch_size;
projected_factors = cell(d, 1);

component_timer = tic;
for mode = 1:d
    projected_factors{mode} = sketch.factors{mode} * X.u{mode};
end
factor_projection_time = toc(component_timer);

component_timer = tic;
values = zeros(sketch_size, 1);
vectors = cell(d, 1);

for row_idx = 1:sketch_size
    for mode = 1:d
        vectors{mode} = projected_factors{mode}(row_idx, :).';
    end
    values(row_idx) = ttv(X.core, vectors);
end
core_contraction_time = toc(component_timer);

info.factor_projection_time_sec = factor_projection_time;
info.core_contraction_time_sec = core_contraction_time;
info.call_time_sec = toc(call_timer);

end
