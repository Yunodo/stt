from fairseq.models.wav2vec import Wav2VecModel

def load_wav2vec(path, map_location):
    cp = torch.load(path, map_location)
    model = Wav2VecModel.build_model(cp['args'], task=None)
    model.load_state_dict(cp['model'], strict = True)
    return model
