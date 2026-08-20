function max_ranks = normalise_tucker_rank_cap( ...
    max_rank, n)
%NORMALISE_TUCKER_RANK_CAP Check and expand a Tucker rank cap.

n = double(n(:).');
d = numel(n);

if d < 1 || any(n < 1) || ...
        any(n ~= floor(n))
    error('n must contain positive integer dimensions.');
end

if isempty(max_rank)
    max_ranks = n;
    return
end

max_ranks = double(max_rank(:).');

if isscalar(max_ranks)
    max_ranks = repmat(max_ranks, 1, d);
elseif numel(max_ranks) ~= d
    error(['max_rank must be empty, scalar, or contain ', ...
           'one entry per tensor mode.']);
end

if any(~isfinite(max_ranks)) || any(max_ranks < 1) || ...
        any(max_ranks ~= floor(max_ranks))
    error('Every maximum Tucker rank must be a positive finite integer.');
end

max_ranks = min(max_ranks, n);

end
