

# image-mosaic

> A Image Mosaic generator made with Python 

## Getting Started

To get up and running, make sure you have [Python](https://www.python.org/) installed

### Prerequisites

This project depends on [Numpy](www.numpy.org/) , [Pillow](https://pillow.readthedocs.io/en/5.1.x/) and [scikit-image](http://scikit-image.org/docs/dev/install.html)

to install them:

Numpy:
```
pip install numpy
```
Pillow:
```
pip install pillow
```
scikit-image
```
pip install scikit-image
```

### User Defined Variables

In the code, there are variables that can be changed

1. The name of the input image -> 'mainImageTarget'

2. Number of tiles to devide image into -> 'numberOfTiles'.
   * For square image you can input 1 number equal to the number of tiles (must be squared number)
   * For non squared images, you input the number of square on x axis, and number of squares on y axis 

3. Comparison Mode -> 
   * [1] Compare transformed rgb to transformed rgb (small tiles)
   * [2] Compare transformed rgb to transformed rgb (big tiles)

4. Transform Mode ->
   * [1] no transformation
   * [2] RGB -> sRGB
   * [3] RGB -> CIE-Lab (my implimentation) 
   * [4] RGB -> CIE-Lab (scikit-image)

*As of the latest version, this is not required, refer to command line usage.*
*However, if you wish to use it the old way, look for this piece of code.*

```python
#& user defined if need be. 
mode = args['mode']						#1
transform = args['transform']			#3
numberOfTiles = args['tiles']			#400
mainImageTarget = args['file']			#"1_in.jpg"
#&
```

## Command-Line Usage

To use the script you need to attached arguments in the command-line,

Refer to above (User Defined Variables) for explination on values, 

+ [-m] mode := comparison mode, is int of range [1 -> 2]
+ [-t] tiles := number of tiles, explained in last section. 
+ [-tr] or [-trans] transform mode := transformation from RGB to ... , is int of range [1 -> 4]
+ [-f] file := target file, is the filename of the image.

Example usage
```
#For square image, 400 tiles (20x20)
$	sudo python python_image_mosaic.py -m 1 -t 200 -tr 3 -f 1_in.jpg

#For non-square image, 3750 tiles (50x75)
$	sudo python python_image_mosaic.py -m 1 -t 50, 75 -tr 3 -f 2_in.jpg
```

## Example Output

Target Image               |  Resulting Image
:-------------------------:|:-------------------------:
![](https://github.com/imranparuk/image-mosaic/blob/master/images/1_in.jpg)  |  ![](https://github.com/imranparuk/image-mosaic/blob/master/images/out_1_in.jpg)

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Imran Paruk** - *Initial work* 

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Thank you to:
+ http://www.easyrgb.com/en/math.php
+ https://stackoverflow.com/a/5021831
+ https://en.wikipedia.org/wiki/CIELAB_color_space#Forward_transformation
+ https://gist.github.com/bikz05/6fd21c812ef6ebac66e1
+ https://gist.github.com/manojpandey/f5ece715132c572c80421febebaf66ae
+ https://stackoverflow.com/a/39504463


