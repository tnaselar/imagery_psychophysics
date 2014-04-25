from PIL import Image
from numpy import random
N=500

im = Image.open('/home/graham/Desktop/Experiment/gwarner/labeled_photos/photos/003030.png')

#rand_array = numpy.reshape(random.random(250000),(500,500)).astype('f8')
rand_array = numpy.random.randint(255, size=(500,500)).astype('uint8')
#rand_array = numpy.reshape(random.random(750000),(750000)).astype('f8')

#rand_pix = tuple(random.randint(0,499, (N,2)).astype('uint64'))
#def coordinate(a,ii):
#	a=1
#	if a==1:
#		return tuple(int(rand_pix[ii,0]),int(rand_pix[ii,1]))

#coordinate = tuple(map(int,tuple(random.randint(0,499, (N,2)).astype('uint64'))))
#coordinate = tuple(int(rand_array[ii,0]),int(rand_array[ii,1]))

rand_image = Image.fromarray(rand_array)

def rand_array_color(rand_image):
	from PIL import Image
	for ii in numpy.nditer(rand_array):
		coordinate = tuple(map(int, rand_image[ii,:]))
		if ii < 128:
			newvalue = rand_image.putpixel(coordinate(a,ii), im.getpixel(coordinate(a,ii)))
 			return newvalue
		else:
			pass

finished_array = rand_array_color(rand_array)

finished_image = Image.fromarray(finished_array)
finished_image.show()

##replace the random values at these pixels with the original values
#N = 500
# def replace(ii,N):
# 	from numpy import random
# 	#coordinate = tuple(map(int, rand_pix[ii,:]))
# 	coordinate = tuple(int(rand_pix[ii,0]),int(rand_pix[ii,1]))
# 	#probability = random.random()
# 	#make tuple of random numbers and apply all at once instead of applying function pixel by pixel
# 	probability = [(random.random() for a in range(500, 500,3))]
# 	for ii in range(N):
# 		if probability < .9:
# 			newvalue = rand_image.putpixel(coordinate, im.getpixel(coordinate))
# 			return newvalue
# 		else:
# 			return ii in range(N)
