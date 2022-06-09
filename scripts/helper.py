import pickle

def read_obj(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def serialize_obj(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


