import albumentations as A
import cv2
import numpy as np

from galaxy_datasets.transforms import base_transforms


class RandomCenteredCrop(A.ImageOnlyTransform):
    def __init__(self, size=(224,224), scale=(0.8, 1.2), rotation=(-180,180), p=1.0, seed=42):
        super().__init__(p=p)
        self.size = size
        self.scale = scale
        self.rotation = rotation
        self.seed = seed
        self.rg = np.random.default_rng(self.seed)
        #print(seed)

    def get_transform_init_args_names(self):
        return ("size", "scale", "rotation", "p", "seed")

    def apply(self, img, **params):
        h,w = img.shape[:2]
        scale = self.rg.uniform(self.scale[0], self.scale[1])  # Random scale factor
        angle = self.rg.uniform(self.rotation[0], self.rotation[1])  # Random rotation

        #print(scale, angle)

        # Compute center of the original image
        center = (w / 2, h / 2)

        # Create affine transformation matrix for scaling and rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)

        # Translate the center to output size/2
        M[0, 2] += self.size[1] / 2 - center[0]
        M[1, 2] += self.size[0] / 2 - center[1]

        # Apply warp transformation (scaling + rotation)
        warped_img = cv2.warpAffine(img, M, (self.size[1],self.size[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        #print(warped_img.shape)
        return warped_img

class KeepCenterOnly(A.ImageOnlyTransform):
    def __init__(self, size=(224,224), radius=40, kernel="disk", p=1.0):
        super().__init__(p=p)
        self.size = size
        self.radius = radius
        self.kernel = kernel

        self.mask = self.generate_mask(radius, kernel)

    def get_transform_init_args_names(self):
        return ("size", "radius", "kernel")

    def generate_mask(self, radius, kernel):
        # Create a grid of coordinates
        y, x = np.indices((self.size[0], self.size[1]))

        # Calculate the center of the image
        center_y, center_x = self.size[0] // 2, self.size[1] // 2

        # Compute the Euclidean distance to the center
        if (self.kernel == "gaussian"):
            sigma = self.radius * 2
            mask = np.exp( - ((x - center_x)**2 + (y - center_y)**2) / 2.0 / sigma**2 )
        elif (self.kernel == "disk"):
            mask = ((x - center_x)**2 + (y - center_y)**2) < (self.radius ** 2)

        return mask

    def apply(self, img, **params):
        return (img * self.mask[:,:,np.newaxis]).astype(np.uint8)


class RemoveAlpha(A.ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)
        # some png images have fourth alpha channel with value of 255 everywhere (i.e. opaque). averaging over this adds incorrect offset

    def get_transform_init_args_names(self):
        return ("p")
    
    def forward(self, img):
        return img[:, :, :3]

    def apply(self, image, **kwargs):
        return self.forward(image)


def custom_transforms(
    crop_scale_bounds=(1.0, 1.0),
    rotation_bounds=(-180.0,180.0),
    resize_after_crop=224,
    to_float=True,  # set to True when loading images directly, False via webdatasets (which normalizes to 0-1 on decode)
    keep_center_only=False
    ) -> A.Compose:

    transforms_to_apply = [RemoveAlpha()]

    transforms_to_apply += [
        RandomCenteredCrop(size=(resize_after_crop,resize_after_crop), scale=crop_scale_bounds, rotation=rotation_bounds),
        A.VerticalFlip(p=0.5),
    ]
    if keep_center_only:
        transforms_to_apply += [KeepCenterOnly(size=(resize_after_crop,resize_after_crop), radius=40)]

    if to_float:
        transforms_to_apply += [A.ToFloat(max_value=255.0)]

    return A.Compose(transforms_to_apply)
