function total = add_tucker_kernel_timing(total, addition)
%ADD_TUCKER_KERNEL_TIMING Add matching leaf-kernel timing structures.

if isempty(addition)
    return
end

timingFields = fieldnames(empty_tucker_kernel_timing());

for fieldIndex = 1:numel(timingFields)
    fieldName = timingFields{fieldIndex};

    if isfield(addition, fieldName)
        total.(fieldName) = total.(fieldName) + addition.(fieldName);
    end
end

end
