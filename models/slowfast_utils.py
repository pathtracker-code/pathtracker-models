try:
    from slowfast.models import video_model_builder
    from slowfast.config.defaults import get_cfg
except:
    print("Failed to import slowfast.")

def slowfast_nl(cfg="models/cfgs/slowfast_nl.yaml"):
    dcfg = get_cfg()
    model = video_model_builder.SlowFast(dcfg)
    return model


def slowfast(cfg="models/cfgs/slowfast.yaml"):
    dcfg = get_cfg()
    dcfg.merge_from_file(cfg)
    model = video_model_builder.SlowFast(dcfg)
    return model


def fast(cfg="models/cfgs/slow.yaml"):
    dcfg = get_cfg()
    dcfg.merge_from_file(cfg)

