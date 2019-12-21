import moviepy.editor as mpy
import uuid
import os
import numpy as np
from PIL import Image

class VideoMaker:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.max_duration = 5000

    def save(self, frames, episode_number=None):
        duration = min(self.max_duration, len(frames))

        def make_filename(episode_number):
            if episode_number is None:
                return str(uuid.uuid4()) + '.mp4'
            else:
                return 'episode_{}.mp4'.format(episode_number)

        def make_frame(t):
            arr = frames[int((len(frames) / duration) * t)]
            arr = (255 * arr).astype(np.int32).astype(np.uint8)
            h,w,c = arr.shape
            img = Image.fromarray(arr)
            img = img.resize(size=(8*w, 8*h), resample=Image.NEAREST)
            arr = np.array(img, dtype=np.float32)
            return arr

        filename = make_filename(episode_number)
        fp = os.path.join(self.output_dir, filename)

        clip = mpy.VideoClip(make_frame, duration=duration)
        _ = clip.write_videofile(fp, fps=1, verbose=False, logger=None)
        return fp