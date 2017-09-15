import sys
from PIL import Image

inFilePath = (sys.argv)[1]

image = Image.open(inFilePath)
data = image.getdata()
newdata = []
outImage = Image.new(image.mode, image.size)
for i in range(len(data)):
	newdata.append((data[i][0]//2,data[i][1]//2,data[i][2]//2))
outImage.putdata(newdata)

outImage.save('Q2.jpg')