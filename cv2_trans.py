from PIL import Image
import cv2
import random
import numpy as np

def random_resize(img, probability, minscale, maxscale):
    rand = random.uniform(minscale, maxscale)
    resize_flag = random.uniform(0, 1)
    if resize_flag > probability:
        img = cv2.resize(img, (int(img.shape[0]*rand), int(img.shape[1]*rand)))
    return img

def random_jpeg(img, probability, minscale, maxscale):
    rand = random.randint(minscale, maxscale)
    jpeg_flag = random.uniform(0, 1)
    decode_img = img
    if jpeg_flag > probability:
        #cv_img = np.array(img)
        encode_img = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), rand])[1]
        decode_img = cv2.imdecode(encode_img, cv2.IMREAD_COLOR)
        #img = Image.fromarray(decode_img)
    return decode_img

def random_blur(img, probability, kernel):
    blur_flag = random.uniform(0, 1)
    #gaussian_kernel = [3, 5, 7, 9]
    gaussian_kernel = kernel
    blur_img = img
    if blur_flag > probability:
        #cv_img = np.array(img)
        k_ind = random.randint(0, len(gaussian_kernel)-1)
        kernel_size = gaussian_kernel[k_ind]
        blur_img = cv2.GaussianBlur(img, ksize=(kernel_size, kernel_size), sigmaX=0, sigmaY=0)
        #img = Image.fromarray(blur_img)
    return blur_img
        
def random_illu(img, probability, illulst):
    illu_flag = random.uniform(0, 1)
    dstimg = img
    if illu_flag < probability:
        rows, cols = img.shape[:2]
        index = random.randint(0,illulst.shape[0]-1)
        mask = illulst[index]
        mask = cv2.resize(mask, (2*cols, 2*rows), interpolation=cv2.INTER_LINEAR)
        # select in left/up/right areas
        randarea = random.randint(0, 2)
        if randarea == 0:  #
            # select center in left area
            shiftX = random.randint(-int(cols/2), -int(cols * 0.4))
            shiftY = random.randint(-int(rows/2), int(rows/2))
        elif randarea == 1:  #
            # select center in up area
            shiftX = random.randint(-int(cols/2), int(cols/2))
            shiftY = random.randint(-int(rows/2), -int(rows * 0.4))
        else:
            # select center in right area
            shiftX = random.randint(int(0.4 * cols), int(cols/2))
            shiftY = random.randint(-int(rows/2), int(rows/2))

        mat_translation = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
        aff_mask = cv2.warpAffine(mask, mat_translation, (2*cols, 2*rows))
        aff_mask = aff_mask[int(rows/2):int(rows/2)+rows, int(cols/2):int(cols/2)+cols]
        dstimg = (img.astype('uint16') + aff_mask[:,:, np.newaxis])
        dstimg = np.clip(dstimg, 0, 255).astype('uint8')
    return dstimg

def random_color(img, probability, colorlst):
    illu_flag = random.uniform(0, 1)
    dstimg = img
    if illu_flag < probability:
        rows, cols = img.shape[:2]
        index = random.randint(0,colorlst.shape[0]-1)
        mask = colorlst[index]
        mask = cv2.resize(mask, (2*cols, 2*rows), interpolation=cv2.INTER_LINEAR)

        shiftX = random.randint(-int(0.05 * cols), int(0.05 * cols))
        shiftY = random.randint(-int(0.05 * rows), int(0.05 * rows))

        mat_translation = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
        aff_mask = cv2.warpAffine(mask, mat_translation, (2*cols, 2*rows))
        aff_mask = aff_mask[int(rows/2):int(rows/2)+rows, int(cols/2):int(cols/2)+cols]
        
        #select 6 combination for three channesl
        randchannel = random.randint(0, 6)
        alpha = random.uniform(0.4,0.6) #alpha*img + mask
        if randchannel == 0: #blue
            dstimg = alpha * img.astype('uint16')[:, :, 0] + (1-alpha)*aff_mask
            dstimg = np.clip(dstimg, 0, 255).astype('uint8')
            dstimg = np.concatenate((dstimg[:, :, np.newaxis], alpha*img[:,:,1:]), axis=2)
        elif randchannel == 1: #green
            dstimg = alpha * img.astype('uint16')[:, :, 1] + (1-alpha)*aff_mask
            dstimg = np.clip(dstimg, 0, 255).astype('uint8')
            dstimg = np.concatenate((alpha*img[:,:,0][:,:,np.newaxis] , dstimg[:, :, np.newaxis], alpha*img[:,:,2][:,:,np.newaxis]), axis=2)
        elif randchannel == 2: #red
            dstimg = alpha * img.astype('uint16')[:, :, 2] + (1-alpha)*aff_mask
            dstimg = np.clip(dstimg, 0, 255).astype('uint8')
            dstimg = np.concatenate((alpha*img[:,:,0:2] , dstimg[:, :, np.newaxis]), axis=2)
        elif randchannel == 3: #blue + green
            dstimg = alpha*img.astype('uint16')[:,:,:2] + (1-alpha)*aff_mask[:,:, np.newaxis]
            dstimg = np.concatenate((np.clip(dstimg, 0, 255).astype('uint8'), alpha*img[:,:,2][:,:,np.newaxis]), axis=2)
        elif randchannel == 4: #blue + red
            dstimg = alpha * img.astype('uint16')[:, :, 0:3:2] + (1-alpha)*aff_mask[:, :, np.newaxis]
            dstimg = np.clip(dstimg, 0, 255).astype('uint8')
            dstimg = np.concatenate((dstimg[:,:,0][:, :, np.newaxis], alpha * img[:, :, 1][:, :, np.newaxis], dstimg[:,:,1][:, :, np.newaxis]), axis=2)
        elif randchannel == 5: #green + red
            dstimg = alpha*img.astype('uint16')[:,:,1:] + (1-alpha)*aff_mask[:,:, np.newaxis]
            dstimg = np.concatenate((alpha*img[:,:,0][:,:,np.newaxis] , np.clip(dstimg, 0, 255).astype('uint8')), axis=2)
        else: #blue + green + red
            dstimg = alpha*img.astype('uint16') + (1-alpha)*aff_mask[:,:, np.newaxis]
        
    return dstimg.astype('uint8')
