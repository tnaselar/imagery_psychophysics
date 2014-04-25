import numpy
import math
import Image
import random
import pdb

im_str = '/musc.repo/Data/katie/testpics4_images/hole_003054.png'
cloud_num = 100
im_size_tuple = (500,500)


im = Image.open(im_str)
pix_list = []
for i in range(im_size_tuple[0]-1):
    for ii in range(im_size_tuple[1]-1):
	if im.getpixel((i,ii)) == (128, 128, 128):
	    pix_list.append((i,ii))
random.shuffle(pix_list)
center = numpy.array(pix_list.pop())
theta = math.radians(random.randint(0,180))
R = numpy.array([math.cos(theta), -1*math.sin(theta), math.sin(theta),math.cos(theta)])
R=R.reshape((2,2))
scale = random.randint(50,100)
cor = random.randint(0, 10)
cov = scale*(numpy.eye(2)+cor)
percent = 0
while percent < 50:
    f_cloud = numpy.random.multivariate_normal(numpy.array((0,0)), cov, cloud_num).dot(R)+numpy.array(center)
    cloud = f_cloud.astype('int')
    percent = 0
    cloud_list = []
    for l in range(len(cloud)):
	cloud_list.append((cloud[l][0],cloud[l][1]))
	if pix_list.count((cloud[l][0],cloud[l][1])) != 0:
	    percent +=1
	print percent
