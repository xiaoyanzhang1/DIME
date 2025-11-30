def get_model(model_name, args):
    name = model_name.lower()
    if name == 'onea':
        from models.onea import Learner
    else:
        assert 0
    return Learner(args)