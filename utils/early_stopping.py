def is_improvement(current_value, best_value, min_delta=0.0):
    return current_value < (best_value - min_delta)


def get_wait_count(current_epoch, best_epoch, start_epoch=0):
    if best_epoch is None:
        return 0
    warmup_boundary = max(start_epoch - 1, 0)
    effective_best_epoch = max(best_epoch, warmup_boundary)
    return max(current_epoch - effective_best_epoch, 0)


def should_stop_early(current_epoch, best_epoch, start_epoch, patience):
    if patience <= 0 or best_epoch is None or current_epoch < start_epoch:
        return False
    return get_wait_count(current_epoch, best_epoch, start_epoch=start_epoch) >= patience
