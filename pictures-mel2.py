import os
import matplotlib

matplotlib.use('Agg')  # No pictures displayed
import pylab
import librosa
import librosa.display
import numpy as np

path = 'wav/03a02Ta.wav'

sig, fs = librosa.load(path)
# make pictures name
save_path = 'test.jpg'

pylab.axis('off')  # no axis
pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
S = librosa.feature.melspectrogram(y=sig, sr=fs)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
pylab.close()