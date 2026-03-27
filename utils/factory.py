def get_model(model_name, args):
    # name = model_name.lower()
    name = model_name
    if name == 'dime':
        from models.dime import Learner
    else:
        assert 0
    return Learner(args)