import imageio
import os
from glob import glob
from PIL import Image
from ying import Ying_2017_CAIP
from dhe import dhe
from he import he
import numpy as np

img_path = glob("archive\\train_another\\damage\\*.jpeg")

# for path in img_path:
#     print(path)
#     (paths,filename)=os.path.split(path)
#     print(paths,filename)
#
#     image = np.array(Image.open(path))
#
#     result = Ying_2017_CAIP(image)
#     result1 = dhe(image)
#     result2 = he(image)
#     imageio.imsave('archive\\train_another_ying\\damage\\'+filename,result)
#     imageio.imsave('archive\\train_another_dhe\\damage\\' + filename, result1)
#     imageio.imsave('archive\\train_another_he\\damage\\' + filename, result2)
#
# img_path = glob("archive\\train_another\\no_damage\\*.jpeg")
# for path in img_path:
#     print(path)
#     (paths,filename)=os.path.split(path)
#     print(paths,filename)
#     image = np.array(Image.open(path))
#     result = Ying_2017_CAIP(image)
#     result1 = dhe(image)
#     result2 = he(image)
#     imageio.imsave('archive\\train_another_ying\\no_damage\\'+filename,result)
#     imageio.imsave('archive\\train_another_dhe\\no_damage\\' + filename, result1)
#     imageio.imsave('archive\\train_another_he\\no_damage\\' + filename, result2)
#
#
# img_path = glob("archive\\test_another\\damage\\*.jpeg")
# for path in img_path:
#     print(path)
#     (paths,filename)=os.path.split(path)
#     print(paths,filename)
#     image = np.array(Image.open(path))
#     result = Ying_2017_CAIP(image)
#     result1 = dhe(image)
#     result2 = he(image)
#     imageio.imsave('archive\\test_another_ying\\damage\\'+filename,result)
#     imageio.imsave('archive\\test_another_dhe\\damage\\' + filename, result1)
#     imageio.imsave('archive\\test_another_he\\damage\\' + filename, result2)
#
# img_path = glob("archive\\test_another\\no_damage\\*.jpeg")
# for path in img_path:
#     print(path)
#     (paths,filename)=os.path.split(path)
#     print(paths,filename)
#
#     image = np.array(Image.open(path))
#     result = Ying_2017_CAIP(image)
#     result1 = dhe(image)
#     result2 = he(image)
#     imageio.imsave('archive\\test_another_ying\\no_damage\\'+filename,result)
#     imageio.imsave('archive\\test_another_dhe\\no_damage\\'+filename,result1)
#     imageio.imsave('archive\\test_another_he\\no_damage\\'+filename,result2)
#
#
# img_path = glob("archive\\test\\damage\\*.jpeg")
# for path in img_path:
#     print(path)
#     (paths,filename)=os.path.split(path)
#     print(paths,filename)
#     image = np.array(Image.open(path))
#     result = Ying_2017_CAIP(image)
#     result1 = dhe(image)
#     result2 = he(image)
#     imageio.imsave('archive\\test_ying\\damage\\'+filename,result)
#     imageio.imsave('archive\\test_dhe\\damage\\'+filename,result1)
#     imageio.imsave('archive\\test_he\\damage\\' + filename, result2)
#
#
# img_path = glob("archive\\test\\no_damage\\*.jpeg")
# for path in img_path:
#     print(path)
#     (paths,filename)=os.path.split(path)
#     print(paths,filename)
#
#     image = np.array(Image.open(path))
#     result = Ying_2017_CAIP(image)
#     result1 = dhe(image)
#     result2 = he(image)
#     imageio.imsave('archive\\test_ying\\no_damage\\'+filename,result)
#     imageio.imsave('archive\\test_dhe\\no_damage\\'+filename,result1)
#     imageio.imsave('archive\\test_he\\no_damage\\'+filename,result2)
#
# img_path = glob("archive\\validation_another\\damage\\*.jpeg")
# for path in img_path:
#     print(path)
#     (paths,filename)=os.path.split(path)
#     print(paths,filename)
#
#     image = np.array(Image.open(path))
#     result = Ying_2017_CAIP(image)
#     result1 = dhe(image)
#     result2 = he(image)
#     imageio.imsave('archive\\validation_another_ying\\damage\\'+filename,result)
#     imageio.imsave('archive\\validation_another_dhe\\damage\\' + filename, result1)
#     imageio.imsave('archive\\validation_another_he\\damage\\' + filename, result2)

img_path = glob("archive\\validation_another\\no_damage\\*.jpeg")
for path in img_path:
    print(path)
    (paths,filename)=os.path.split(path)
    print(paths,filename)

    image = np.array(Image.open(path))
    result = Ying_2017_CAIP(image)
    result1 = dhe(image)
    result2 = he(image)
    imageio.imsave('archive\\validation_another_ying\\no_damage\\'+filename,result)
    imageio.imsave('archive\\validation_another_dhe\\no_damage\\' + filename, result1)
    imageio.imsave('archive\\validation_another_he\\no_damage\\' + filename, result2)