import os
from skimage import io, util

SRCDIR='/path/to/modelnet40v1png'
TARGETDIR='/path/to/modelnet40v1'

dirs = os.listdir(SRCDIR)
phase = ('train', 'test')
if not os.path.exists(TARGETDIR):
    os.makedirs(TARGETDIR)

for d in dirs:
    print(d)
    for p in phase:
        path = os.path.join(SRCDIR, d, p)
        target_path = os.path.join(TARGETDIR, d, p)
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        imgs = [f for f in os.listdir(path) if os.path.splitext(f)[1] == '.png']

        for im_name in imgs:
            img = io.imread(os.path.join(path, im_name))
            img = util.pad(img, ((16, 16), (16, 16), (0, 0)), 'edge')
            io.imsave(os.path.join(target_path, im_name), img)
