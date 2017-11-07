# chainer-PGGAN
 Progressive Growing of GANs implemented with chainer

 `python 3.5.2` + `chainer 3.0.0`

## Usage

### Training

```
$ python3 ./train.py -g 0 --dir ./train_images/ --epoch 100 --depth 0 
```

You can train models with `./train.py`.

When `depth = n`, generated images are `2^{n+2} x 2^{n+2}` size.

```
$ ./batch.sh 100
```

`batch.sh` automatically trains models gradually (through `4 x 4` to `256 x 256`).

You should tune `delta` and `epoch` when it changes too quickly or too slowly.

### Generating

```
$ python3 ./analogy.py --gen results/gen --depth 0
```

You can generate images with `analogy.py`

```
$ wget https://www.dropbox.com/s/dvnxb4vur6fasei/gen_yui_model
$ python3 ./analogy.py --gen gen_yui_model --depth 6
```

You can use the pre-trained model.

It generates `256 x 256` size Ichii Yui's images.

## Bibliography

[1] http://research.nvidia.com/publication/2017-10_Progressive-Growing-of

The original paper

[2] https://github.com/dhgrs/chainer-WGAN-GP

WGAN-GP implemented with chainer.

[3] http://joisino.hatenablog.com/entry/2017/11/07/200000

My Blog post related to this repository.
