function total = add_tucker_kernel_timing(total, addition)
%ADD_TUCKER_KERNEL_TIMING Add matching leaf-kernel timing structures.

if isempty(addition)
    return
end

timing_fields = fieldnames(empty_tucker_kernel_timing());

for field_idx = 1:numel(timing_fields)
    field_name = timing_fields{field_idx};

    if isfield(addition, field_name)
        total.(field_name) = total.(field_name) + addition.(field_name);
    end
end

end
