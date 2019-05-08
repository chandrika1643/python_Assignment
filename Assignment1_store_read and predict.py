### IMPORTING MODULES
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,metrics
from sklearn.ensemble import RandomForestClassifier
from scipy import misc
from numpy import empty
### Reading and storing an image from DATABASE and EXTERNAL
# the digits dataset 
digits=datasets.load_digits()
t=digits.target
#the image to predict_(testing)#the size,datatype and the values/pixels 
#are same as the trained set since the image is taken from the same set,
#othercase the attributes to be matched
im=digits.images[16]
ext_im=cv2.imread('5_five.png')

plt.subplot(1,2,1)
plt.axis('off')
plt.imshow(ext_im,cmap='gray',interpolation='nearest')
plt.title('External image')
plt.show()

plt.subplot(1,2,2)
plt.axis('off')
plt.imshow(im,interpolation='nearest')
plt.title('database image')
plt.show()

print('image from database and its shape and datatype')
plt.gray()
plt.matshow(im)
plt.show()
print(im.shape)
print(im.dtype)
im2=im.reshape(1,64)
print('size after reshaping',im2)

#processing on external image
print('External image and its shape and datatype')
print(ext_im.shape)
print(ext_im.dtype)
ext_im2=cv2.resize(ext_im,(8,8))#resizesto 8*8
plt.imshow(ext_im2)
print('reshaped to ',ext_im2.shape)
ext_im3=np.asarray(ext_im2,dtype='float64')#converts the datatype
print('datatype converted to',ext_im3.dtype)
ext_im4=misc.bytescale(ext_im3,high=16,low=0)
print('scales pixels from 0 to 255 to 0 to 16 \n',ext_im4)
final=[]
for row in ext_im4:
    for pixel in row:
        final.append(sum(pixel)/3.0) 
final=np.array(final)
ext_im5=final.reshape(1,64)
print('1x64 matrix',ext_im5)

print(digits.images.shape)
n_samples=len(digits.images)
data=digits.images.reshape((n_samples,-1))#flattening the test image
print(data)
data.shape

### TRAINING OF THE ALGORITHM and prediction

#create a classifiee :a random forest classifier
classifier=RandomForestClassifier(n_estimators=30,criterion='entropy')
#we learn the digits on the first half of the digits
classifier.fit(data,digits.target)

d_list=[]
d_list=[im2,ext_im5]#list to be predicted
i_list=[im,ext_im]# images under test

im_pd=list(zip(d_list,i_list))
for i,(val,img) in enumerate(im_pd[:2]):
    value=classifier.predict(val)
    print('predicted value \n',value)
    plt.subplot(2,4,i+1)
    plt.imshow(img,interpolation='nearest')
    plt.axis('off')
    plt.show()