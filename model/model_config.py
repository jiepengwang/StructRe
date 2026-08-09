from model.part_vae.part_vae import PartVAE
from model.structmapping.structmapping import StructMapping


def create_model(args, conf):
    arch_model = conf['model.arch']
    if arch_model == 'sparse_vae':
        from model.sparse_vae.sparse_vae import SparseVAE
        model = SparseVAE(args, conf)
    elif arch_model == 'part_vae':
        model = PartVAE(args, conf)
    elif arch_model == 'structmapping':
        model = StructMapping(args, conf)
    else:
        raise ValueError(f"Model [{arch_model}] not recognized.")

    print("[*] Model has been created: %s" % arch_model)
    model.cuda()
    return model
