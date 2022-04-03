# -*- coding: utf-8 -*-
"""pets_classification.ipynb

If you're using a computer with an unusually small GPU, you may get an out of memory error when running this notebook. 
If this happens, click Kernel->Restart, uncomment the 2nd line below to use a smaller *batch size* (you'll learn all about what this means during the course), and try again. 
## Looking at the data

We are going to use the [Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) by [O. M. Parkhi et al., 2012](http://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf) 
which features 12 cat breeds and 25 dogs breeds. Our model will need to learn to differentiate between these 37 distinct categories. 
According to their paper, the best accuracy they could get in 2012 was 59.21%, using a complex model that was specific to pet detection, with separate "Image", "Head", and "Body" models for the pet photos. 
Let's see how accurate we can be using deep learning!

The main difference between the handling of image classification datasets is the way labels are stored. In this particular dataset, labels are stored in the filenames themselves. 
We will need to extract them to be able to classify the images into the correct categories. Fortunately, the fastai library has a handy function made exactly for this, 
`ImageDataBunch.from_name_re` gets the labels from the filenames using a [regular expression](https://docs.python.org/3.6/library/re.html).
Set the random seed to two to guarantee that the same validation set is every time. This will give you consistent results with what you see in the lesson video.

Basically, resnet50 usually performs better because it is a deeper network with more parameters. Let's see if we can achieve a higher performance here. To help it along, let's use larger images too, since that way the network can see more detail. We reduce the batch size a bit since otherwise this larger network will require more GPU memory.

It's astonishing that it's possible to recognize pet breeds so accurately! Let's see if full fine-tuning helps:"""



from fastai.vision import *
from fastai.metrics import error_rate
import warnings
warnings.filterwarnings("ignore")

#  bs = 64
bs = 16   # uncomment this line if you run out of memory 
#help(untar_data)

# `untar_data` function to extract and download data
path = untar_data(URLs.PETS); path
path.ls()

path_anno = path/'annotations'
path_img = path/'images'

fnames = get_image_files(path_img)
print(fnames[:5])

np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs
                                  ).normalize(imagenet_stats)
# random images from dataset
data.show_batch(rows=3, figsize=(7,6))

print(data.classes)
print(len(data.classes),data.c)

learn = cnn_learner(data, models.resnet34, metrics=error_rate)
print(learn.model)

## Training: resnet34
learn.fit_one_cycle(4)
learn.save('/your/path/to file')

interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()

print(len(data.valid_ds)==len(losses)==len(idxs))

## Results
interp.plot_top_losses(9, figsize=(15,11))
doc(interp.plot_top_losses)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

interp.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1');

learn.lr_find()
learn.recorder.plot()

## Unfreezing, fine-tuning, and learning rates
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-5))

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),
                                   size=299, bs=bs//2).normalize(imagenet_stats)

## Training: resnet50
learn = cnn_learner(data, models.resnet50, metrics=error_rate)

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(8)
learn.save('/your/path/to file1')

learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))

learn.load('to file1');

interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)

path = untar_data(URLs.MNIST_SAMPLE); path
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26)

data.show_batch(rows=3, figsize=(5,5))

learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit(2)

df = pd.read_csv(path/'labels.csv')
df.head()

data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=28)

data.show_batch(rows=3, figsize=(5,5))
data.classes

data = ImageDataBunch.from_df(path, df, ds_tfms=tfms, size=24)
data.classes

fn_paths = [path/name for name in df['name']]; fn_paths[:2]

pat = r"/(\d)/\d+\.png$"
data = ImageDataBunch.from_name_re(path, fn_paths, pat=pat, ds_tfms=tfms, size=24)
data.classes

data = ImageDataBunch.from_name_func(path, fn_paths, ds_tfms=tfms, size=24,
        label_func = lambda x: '3' if '/3/' in str(x) else '7')
data.classes

labels = [('3' if '/3/' in str(x) else '7') for x in fn_paths]
labels[:5]

data = ImageDataBunch.from_lists(path, fn_paths, labels=labels, ds_tfms=tfms, size=24)
data.classes

