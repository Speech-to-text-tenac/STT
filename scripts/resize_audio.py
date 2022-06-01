import librosa


def resize_audio(audios:dict,max_duration:float) -> dict:
    """Extend duration of audio samples to max_duration.

    Args:
        audios (dict): Dictionary of audio samples.
        max_duration (float): The duration to set for the audio samples

    Returns:
        dict: Dictionary of resized audio samples.
    """
    resized_audios = {}
    for label in audios:
        resized_audios[label] = librosa.util.fix_length(audios[label],size=int(max_duration*44100))
    return resized_audios