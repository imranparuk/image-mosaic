

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

### Run

To run:

```
python python_image_mosaic.py
```

### User Defined Variables

In the code, there are variables that can be changed
1) The name of the input image -> 'mainImageTarget'
2) Number of tiles to devide image into -> 'numberOfTiles'
3) Mode -> [1] Compare to rgb to rgb (small resized images)
		   [2] Compare CIE-lab to CIE-lab (small resized images)
		   [3] Compare to rgb to rgb (original size images)
		   [4] Compare CIE-lab to CIE-lab (original size images)
4) Use library for the RGB->CIE-lab conversion (True = yes, False = no)

Change in this section of the code:
```python
##& -> User Defined Variables
mainImageTarget = "1_in.jpg"
numberOfTiles = 1600
mode = 2    
useSkImage = False
##&
```
## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Imran Paruk** - *Initial work* 

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Thank you to:

https://gist.github.com/bikz05/6fd21c812ef6ebac66e1
https://gist.github.com/manojpandey/f5ece715132c572c80421febebaf66ae
