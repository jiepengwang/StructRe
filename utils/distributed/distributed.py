from torch import distributed as dist


LOCAL_PROCESS_GROUP = None


def get_local_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0

    if LOCAL_PROCESS_GROUP is None:
        raise ValueError("utils.distributed.LOCAL_PROCESS_GROUP is None")

    return dist.get_rank(group=LOCAL_PROCESS_GROUP)
