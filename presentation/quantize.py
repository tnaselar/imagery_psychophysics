from PIL import Image
rootdir = '/Users/thomasnaselaris/Data/tmp/' ## put root directory here
masks = rootdir+'labeled_photos/masks/' ##location of masks
photo = rootdir+'labeled_photos/photos/' ##locations of photos
savedir = rootdir+'quant/%s/' ##place where you want your files saved.
q_threshold = 127


def quantize(im, T):
	'''
	Quantize an image
	Input:
		im ~ a image object
		 T ~ threshold pixel value between 0 and 255 
	Output:
		A quantized version of the input imgage. All pixels with value > T go to 1,
		all pixels with value <= T go to 0
	'''
	im = im.convert(mode="L")
	thresh = lambda x,y: x > y
	im = im.point(lambda x: 255*thresh(x,y=T))
	return im

##helper for selecting pixels with a specific value
def select(x,y):
	return x==y

##return a bounding box around a specific object in an object mask
def object_box(mask,lev):
	return mask.point(lambda x: 255*select(x,lev)).getbbox()

##not sure if we really need this...
def paste_from_box(im1,im2,box):
	return im1.paste(im2,box)

##check this out: you can convert a image object into a numpy array.
##I found out you can also convert from arry to image object using "Image.fromarray"
def find_object_values(mask):
	'''
	object_values = find_object_values(mask)
	Find pixel values of each of the objects in an object mask
	Returns a numpy array containing the values. 
	len(object_values) = total number of objects in the mask
	'''
	from numpy import unique, array
	return unique(array(mask))

##do individual quantized objects in a box look like anything?
##do random quantized patches look like anything after you've seen them in context?
##can you blur the shit out of quantized images to make them unrecognizable?

for ii in range(3030,3107):
   f='%0.6d.png' %(ii)
   pic = Image.open(photo+f)
   mask =  Image.open(masks+f)
   quantize(pic, q_threshold).save((savedir+f) %('photos'))
   object_vals = find_object_values(mask)
   print 'grabbing objects from photo %06d' %ii
   for vv in object_vals:
   	quantize(pic.crop(object_box(mask,vv)), q_threshold).save((savedir+'image_%0.6d_object_%03d.png') %('objects', ii,vv))
   
