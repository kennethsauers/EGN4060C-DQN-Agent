from array2gif import write_gif
import os

GIF_PATH = 'gif/'

if not os.path.exists(GIF_PATH):
    os.makedirs(GIF_PATH)

class GifMaker():
    def __init__(self):
        self.imageList = []
        self.fps = 30

    def log(self, img):
        self.imageList.append(img)

    def makeGif(self, filename = 'testing', removeImageList = True):
        write_gif(self.imageList, GIF_PATH + filename + '.gif', fps= self.fps)
        if removeImageList:
            self.imageList = []
