import PIL, PIL.ImageOps, PIL.ImageDraw
from PIL import Image
import numpy as np
import random
import torch
from torchvision import transforms

FILL_MEAN = False
FILL_COLOR = (0, 0, 0)
INTERPOLATION = Image.BICUBIC

class ReflectionPaddingFunctor(object):
    def __init__(self, transform, image_size=51):
        self.transform = transform
        self.pad = image_size - 1
        self.crop = (self.pad, self.pad, self.pad + image_size, self.pad + image_size)
        self.__name__ = transform.__name__
    
    def __call__(self, img, val):
        padded_img = transforms.ToPILImage()(torch.nn.ReflectionPad2d(self.pad)(transforms.ToTensor()(img).unsqueeze(0))[0])
        transformed_img = self.transform(padded_img, val)
        return transformed_img.crop(self.crop)
    

def AutoContrast(img, v):
    return PIL.ImageOps.autocontrast(img, v)

def Posterize(img, v):
    v = int(v)
    return PIL.ImageOps.posterize(img, v)

def ShearX(img, v, image_size=51):  # [-0.3, 0.3]
    if random.random() > 0.5:
        v = -v
    flipped = False
    if random.random() > 0.5:
        flipped = True
        img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    if FILL_MEAN:
        fillcolor = tuple([int(x) for x in PIL.ImageStat.Stat(img).mean])
    else:
        fillcolor = FILL_COLOR

        
    img = img.transform(img.size, PIL.Image.AFFINE, (1, v, -np.sign(v) * (image_size - 1), 0, 1, (image_size - 1)), resample=INTERPOLATION, fillcolor=(0, 0, 0))
    img = img.transform(img.size, PIL.Image.AFFINE, (1, 0, np.sign(v) * (image_size - 1), 0, 1, -(image_size - 1)), resample=0, fillcolor=(0, 0, 0))
    if flipped:
        img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    return img

def ShearY(img, v, image_size=51):  # [-0.3, 0.3]
    if random.random() > 0.5:
        v = -v
    flipped = False
    if random.random() > 0.5:
        flipped = True
        img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    if FILL_MEAN:
        fillcolor = tuple([int(x) for x in PIL.ImageStat.Stat(img).mean])
    else:
        fillcolor = FILL_COLOR
    img = img.transform(img.size, PIL.Image.AFFINE, (1, 0, (image_size - 1), v, 1, -np.sign(v) * (image_size - 1)), resample=INTERPOLATION, fillcolor=(0, 0, 0))
    img = img.transform(img.size, PIL.Image.AFFINE, (1, 0, -(image_size - 1), 0, 1, np.sign(v) * (image_size - 1)), resample=0, fillcolor=(0, 0, 0))
    if flipped:
        img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    return img


def TranslateX(img, v, image_size=51):  # [-150, 150] => percentage: [-0.45, 0.45]
    if random.random() > 0.5:
        v = -v
    v = v * image_size
    if FILL_MEAN:
        fillcolor = tuple([int(x) for x in PIL.ImageStat.Stat(img).mean])
    else:
        fillcolor = FILL_COLOR
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0), fillcolor=fillcolor)


def TranslateY(img, v, image_size=51):  # [-150, 150] => percentage: [-0.45, 0.45]
    if random.random() > 0.5:
        v = -v
    v = v * image_size
    if FILL_MEAN:
        fillcolor = tuple([int(x) for x in PIL.ImageStat.Stat(img).mean])
    else:
        fillcolor = FILL_COLOR
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v), fillcolor=fillcolor)

def Rotate(img, v):  # [-30, 30]
    if random.random() > 0.5:
        v = -v
    if FILL_MEAN:
        fillcolor = tuple([int(x) for x in PIL.ImageStat.Stat(img).mean])
    else:
        fillcolor = FILL_COLOR
    return img.rotate(v, INTERPOLATION, fillcolor=fillcolor)

def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)

def SolarizeAdd(img, v):
    return Image.blend(img, Solarize(img, v), 0.5)

def Contrast(img, v):  # [0, 0.9]
    return PIL.ImageEnhance.Contrast(img).enhance(1 + v)

def Color(img, v):  # [0, 0.9]
    return PIL.ImageEnhance.Color(img).enhance(1 + v)

def Brightness(img, v):  # [0, 0.9]
    return PIL.ImageEnhance.Brightness(img).enhance(1 + v)

def Sharpness(img, v):  # [0, 0.9]
    return PIL.ImageEnhance.Sharpness(img).enhance(1 + v)

def Identity(img, _):
    return img

def Cutout(img, v, fcolor=None):  # [0, 60] => percentage: [0, 0.2]
    if v <= 0.:
        return img
    v = v * img.size[0]
    return CutoutAbs(img, v, fcolor)

def CutoutAbs(img, v, fcolor=None):  # [0, 60] => percentage: [0, 0.2]
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    if FILL_MEAN:
        color = tuple([int(x) for x in PIL.ImageStat.Stat(img).mean])
    else:
        color = FILL_COLOR
    if fcolor is not None:
        color = fcolor
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

def Invert(img, _):
    return PIL.ImageOps.invert(img)

def Equalize(img, _):
    return PIL.ImageOps.equalize(img)

def augment_list(im_size):  # 16 oeprations and their ranges
    l = [
        (Identity, 0., 1.0), # 0
        (ReflectionPaddingFunctor(ShearX, image_size=im_size), 0., 0.5),  # 1
        (ReflectionPaddingFunctor(ShearY, image_size=im_size), 0., 0.5),  # 2
        (ReflectionPaddingFunctor(TranslateX, image_size=im_size), 0., 0.45),  # 3
        (ReflectionPaddingFunctor(TranslateY, image_size=im_size), 0., 0.45),  # 4
        (ReflectionPaddingFunctor(Rotate, image_size=im_size), 0, 40),  # 5
        (AutoContrast, 0, 10),  # 6
        (Solarize, 256, 128),  # 7
        (SolarizeAdd, 256, 128),  # 8
        (Posterize, 8, 2),  # 9
        (Contrast, 0, 0.8),  # 10
        (Color, 0, 0.9),  # 11
        (Brightness, 0, 0.8),  # 12
        (Sharpness, 0, 0.9),  # 13
        (Cutout, 0, 0.5),  # 14
    ]
    return l


class BetterRandAugment:
    def __init__(self, n, m, rand_m=False, resample=True, verbose=False, transform=None, true_m0=False, randomize_sign=True, used_transforms=None, image_size=51):
        self.n = n
        self.m = m
        self.max_magnitude = m
        self.rand_m = rand_m
        self.augment_list = augment_list(image_size)
        self.verbose = verbose
        self.true_m0 = true_m0
        self.randomize_sign = randomize_sign
        if used_transforms is None:
            self.used_transforms = list(range(len(augment_list(image_size))))
        else:
            self.used_transforms = used_transforms
        
        self.resample = resample
        if transform is None:
            self.resample_transform()
        elif isinstance(transform, str):
            self.set_transform_str(transform)
        else:
            self.set_transform(transform)
    
    def resample_transform(self):
        self.op_inds = random.choices(self.used_transforms, k=self.n)
        self.ops = [self.augment_list[i] for i in self.op_inds]
        if self.rand_m:
            self.Ms = np.random.uniform(-self.m, self.m, self.n)
        else:
            self.Ms = [self.m for _ in range(self.n)]
        if self.verbose:
            print('Resampled transform. Current transform: ')
            print(str(self.get_transform_str()))
            operation_names = [(self.augment_list[ind][0].__name__, m) for ind, m in zip(self.op_inds, self.Ms)]
            print(f"Resampled transform: {operation_names}")
    
    def set_transform(self, transform):
        self.op_inds = []
        self.Ms = []
        for ind, m in transform:
            self.op_inds.append(ind)
            self.Ms.append(m)
        self.ops = [self.augment_list[i] for i in self.op_inds]
        if self.verbose:
            print('Manually set transform. Current transform: ')
            print(str(self.get_transform_str()))
    
    def get_transform(self):
        transform = []
        for ind, m in zip(self.op_inds, self.Ms):
            transform.append((ind, m))
        return transform
    
    def get_transform_str(self):
        return ''.join(str(self.get_transform()).split())
    
    def set_transform_str(self, s):
        return self.set_transform(eval(s))

    def __call__(self, img):
        if self.resample:
            if self.verbose:
                print('Updated transform!')
            self.resample_transform()
        for i, (op, minval, maxval) in enumerate(self.ops):
            m = self.Ms[i]
            if np.abs(m) < 0.5 and self.true_m0:
                continue
            if self.randomize_sign:
                if np.random.randn() < 0.5:
                    m *= -1
            if op.__name__ not in ['Contrast', 'Color', 'Brightness', 'Sharpness']:
                m = np.abs(m)
            val = (m / self.max_magnitude) * float(maxval - minval) + minval
            img = op(img, val)
        return img
