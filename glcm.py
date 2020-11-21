import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
import cv2
import imagehelper

"""
그래프를 그린다.
"""
def show_patches(image_path, salt_patches, non_salt_patches):
    xs = []
    ys = []
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)

    for patch in (non_salt_patches + salt_patches):
        glcm = greycomatrix(patch, distances=[1], angles=[90], levels=256,
                            symmetric=True, normed=True)
        xs.append(greycoprops(glcm, 'ASM')[0, 0])
        ys.append(greycoprops(glcm, 'correlation')[0, 0])

    # create the figure
    fig = plt.figure(figsize=(8, 8))

    # display original image with locations of patches
    ax = fig.add_subplot(3, 2, 1)
    ax.imshow(image, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    # for (y, x) in grass_locations:
    #    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
    # for (y, x) in sky_locations:
    #    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
    ax.set_xlabel('Original Image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')

    # for each patch, plot (dissimilarity, correlation)
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(xs[:len(non_salt_patches)], ys[:len(non_salt_patches)], 'go',
            label='non-salt')
    ax.plot(xs[len(non_salt_patches):], ys[len(non_salt_patches):], 'bo',
            label='salt')
    ax.set_xlabel('GLCM Dissimilarity')
    ax.set_ylabel('GLCM Correlation')
    ax.legend()

    # display the patches and plot
    fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()

"""
glcm을 가져온다.
return non_salt_patches, salt_patches
"""
def treat_glcm(image_path, mask_path, PATCH_SIZE = 31):

    # open the camera image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)
    # select some patches from grassy areas of the image
    non_salt_locations=[]
    salt_locations=[]
    for i1 in range(int(image.shape[0]/5)):
        for j1 in range(int(image.shape[1]/5)):
            i=i1*5
            j=j1*5
            if(image2[i,j]==0):
                non_salt_locations.append((i,j))
            if(image2[i,j]==255):
                salt_locations.append((i,j))
    non_salt_patches = []
    for loc in non_salt_locations:
        non_salt_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                   loc[1]:loc[1] + PATCH_SIZE])

    # select some patches from sky areas of the image
    salt_patches = []
    for loc in salt_locations:
        salt_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                 loc[1]:loc[1] + PATCH_SIZE])
    return non_salt_patches, salt_patches
