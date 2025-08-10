import numpy as np
import kornia
import torch
import torchvision.transforms.functional as TF

from torch import Tensor
from torchvision import transforms
from collections import OrderedDict

class NormalizeInverse(transforms.Normalize):
    """
    Normalize를 되돌리는 클래스(z-score 표준화 된 이미지를 원래 이미지로 되돌림)
    원본 텐서가 변경되지 않도록 clone() 사용
    Source: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    """

    def __init__(self, mean, std):
        '''
        params:
            mean (Tensor): 평균값
            std (Tensor): 표준편차
        '''
        std_inverse = 1 / (std + 1e-7)
        mean_inverse = -mean * std_inverse
        super().__init__(mean=mean_inverse, std=std_inverse)

    def __call__(self, tensor):
        '''
        params:
            tensor (Tensor): 정규화된 텐서
        return:
            Tensor: 역정규화된 텐서
        '''
        return super().__call__(tensor.clone())


class CropDict:
    """ 
    이미지를 잘라내는 클래스
    start_x, start_y, end_x, end_y 좌표를 기준으로 이미지를 잘라냄
    src 파라미터를 통해 기준 이미지를 정하고 그 이미지를 기준으로 잘라냄
        - src 이미지에서 잘라낸 이미지의 비율에 맞게 나머지 이미지를 잘라냄
    """

    def __init__(self, start_x, start_y, end_x, end_y, src="lr"):
        """
        start_x : int
            자르기를 시작할 x 좌표
        start_y : int
            자르기를 시작할 y 좌표
        end_x : int
            자르기를 끝낼 x 좌표
        end_y : int
            자르기를 끝낼 y 좌표
        src : str, optional
            원본/소스 이미지의 종류, 기본값은 'lr'
        """

        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        self.src = src

    def crop(self, item):
        """
        params:
            item (dict[str, Tensor] 또는 Tensor): 자를 딕셔너리 아이템
        return:
            dict ([str, Tensor] 또는 Tensor): 잘린 딕셔너리 아이템
        """
        start_x, start_y, end_x, end_y = (
            self.start_x,
            self.start_y,
            self.end_x,
            self.end_y,
        )
        if isinstance(item, dict):
            source_img = item[self.src]
            source_height, source_width = source_img.shape[-2:]
            for image_type, image in item.items():
                if isinstance(image, Tensor):
                    image_height, image_width = image.shape[-2:]

                    ratio_height, ratio_width = (
                        image_height / source_height,
                        image_width / source_width,
                    )

                    crop_start_x = round(start_x * ratio_width)
                    crop_start_y = round(start_y * ratio_height)
                    crop_end_x = round(end_x * ratio_width)
                    crop_end_y = round(end_y * ratio_height)

                    item[image_type] = TF.crop(
                        img=image,
                        top=crop_start_y,
                        left=crop_start_x,
                        height=crop_end_x,
                        width=crop_end_y,
                    ).clone()

        elif isinstance(item, Tensor):
            item = TF.crop(item, start_y, start_x, end_y, end_x).clone()
        return item

    # Disable gradients for effiency.
    @torch.no_grad()
    def __call__(self, item):
        return self.crop(item)


class RandomCropDict:
    '''
    이미지의 무작위 위치에 대해 지정된 크기(size)로 자르는 클래스
    src 파라미터를 통해 기준 이미지를 정하고 그 이미지를 기준으로 잘라냄
        - src 이미지에서 잘라낸 이미지의 비율에 맞게 나머지 이미지를 잘라냄
    '''
    def __init__(self, src, size, batched=False):
        '''
        params:
            src (str): 원본/소스 이미지의 종류
            size (tuple of int): 자를 크기
            batched (bool, optional): 배치 여부
                - 배치 차원이 있으면 배치 차원을 제거하고 작업을 수행한 후 다시 배치 차원을 추가하기 위함(e.g., (B, 8, H, W) -> (B*8, H, W))
        '''
        self.src = src
        self.size = size
        self.batched = batched

    def random_crop(self, item):

        if isinstance(item, dict):
            source_image = item[self.src]
            height, width = source_image.shape[-2:]
            random_crop = transforms.RandomCrop.get_params(
                source_image, output_size=self.size
            )

            source_crop_start_y, source_crop_start_x = random_crop[:2]
            source_crop_end_y, source_crop_end_x = random_crop[2:]

            for image_type, image in item.items():
                if isinstance(image, Tensor):

                    if self.batched and image.ndim == 5:
                        batch_size, revisits = image.shape[:2]
                        image = image.flatten(start_dim=0, end_dim=1)

                    image_height, image_width = image.shape[-2:]
                    ratio_height, ratio_width = (
                        image_height / height,
                        image_width / width,
                    )

                    crop_start_y = round(source_crop_start_y * ratio_height)
                    crop_start_x = round(source_crop_start_x * ratio_width)
                    crop_end_y = round(source_crop_end_y * ratio_height)
                    crop_end_x = round(source_crop_end_x * ratio_width)

                    item[image_type] = TF.crop(
                        img=image,
                        top=crop_start_y,
                        left=crop_start_x,
                        height=crop_end_x,
                        width=crop_end_y,
                    ).clone()

                    if self.batched:
                        item[image_type] = item[image_type].unflatten(
                            dim=0, sizes=(batch_size, revisits)
                        )

        elif isinstance(item, Tensor):
            random_crop = transforms.RandomCrop.get_params(
                source_image, output_size=self.size
            )

            source_crop_start_y, source_crop_start_x = random_crop[:2]
            source_crop_end_y, source_crop_end_x = random_crop[2:]

            item[image_type] = TF.crop(
                img=item,
                top=crop_start_y,
                left=crop_start_x,
                height=crop_end_x,
                width=crop_end_y,
            ).clone()

        return item

    @torch.no_grad()
    def __call__(self, item: dict) -> dict:
        return self.random_crop(item)


class RandomRotateFlipDict:
    '''
    딕셔너리 아이템을 무작위 각도로 회전하고 좌우 또는 상하로 뒤집는 클래스
    angles 파라미터를 통해 회전할 각도를 정하고 그 각도에 맞게 회전하고 뒤집음
    '''
    def __init__(self, angles, batched: bool = False):
        '''
        params:
            angles (list of int): 회전할 각도
            batched (bool, optional): 배치 여부
        '''
        self.angles = angles
        self.batched = batched

    @torch.no_grad()
    def __call__(self, item):

        random_angle_index = torch.randint(4, size=(1,)).item()
        angle = self.angles[random_angle_index]
        flip = torch.randint(2, size=(1,)).item()
        if isinstance(item, dict):
            for image_type, image in item.items():
                if isinstance(image, Tensor):
                    if self.batched and image.ndim == 5:
                        batch_size, revisits = image.shape[:2]
                        image = image.flatten(start_dim=0, end_dim=1)
                    item[image_type] = TF.rotate(image, angle)
                    if flip:
                        item[image_type] = TF.vflip(item[image_type])
                    if self.batched:
                        item[image_type] = item[image_type].unflatten(
                            dim=0, sizes=(batch_size, revisits)
                        )

        elif isinstance(item, Tensor):
            item = TF.rotate(item, angle)
            if flip:
                item = TF.vflip(item)
        return item
    
def lanczos_kernel(translation_in_px, kernel_lobes=3, kernel_width=None):
    """ Generates 1D Lanczos kernels for translation and interpolation.
    Adapted from: https://github.com/ElementAI/HighRes-net/blob/master/src/lanczos.py

    Parameters
    ----------
    translation_in_px : Tensor
        Translation in (sub-)pixels, (B,1).
    kernel_lobes : int, optional
        Number of kernel lobes, by default 3.
        If kernel_lobes is None, then the width is the kernel support 
        (length of all lobes), S = 2(a + ceil(subpixel_x)) + 1.
    kernel_width : Optional[int], optional
        Kernel width, by default None.

    Returns
    -------
    Tensor
        1D Lanczos kernel, (B,) or (N,) or (S,).
    """

    device = translation_in_px.device
    dtype = translation_in_px.dtype

    absolute_rounded_translation_in_px = translation_in_px.abs().ceil().int()
    # width of kernel support
    kernel_support_width = 2 * (kernel_lobes + absolute_rounded_translation_in_px) + 1

    maximum_support_width = (
        kernel_support_width.max()
        if hasattr(kernel_support_width, "shape")
        else kernel_support_width
    )

    if (kernel_width is None) or (kernel_width < maximum_support_width):
        kernel_width = kernel_support_width

    # Width of zeros beyond kernel support
    zeros_beyond_support_width = (
        ((kernel_width - kernel_support_width) / 2).floor().int()
    )

    start = (
        -(
            kernel_lobes
            + absolute_rounded_translation_in_px
            + zeros_beyond_support_width
        )
    ).min()
    end = (
        kernel_lobes
        + absolute_rounded_translation_in_px
        + zeros_beyond_support_width
        + 1
    ).max()
    x = (
        torch.arange(start, end, dtype=dtype, device=device).view(1, -1)
        - translation_in_px
    )
    px = (np.pi * x) + 1e-3

    sin_px = torch.sin(px)
    sin_pxa = torch.sin(px / kernel_lobes)

    # sinc(x) masked by sinc(x/a)
    k = kernel_lobes * sin_px * sin_pxa / px ** 2

    return k


def lanczos_shift(x, shift, padding=3, kernel_lobes=3):
    """ Shifts an image by convolving it with a Lanczos kernel.
    Lanczos interpolation is an approximation to ideal sinc interpolation,
    by windowing a sinc kernel with another sinc function extending up to
    a few number of its lobes (typically 3).

    Adapted from:
            https://github.com/ElementAI/HighRes-net/blob/master/src/lanczos.py

    Parameters
    ----------
    x : Tensor
        Image to be shifted, (batch_size, channels, height, width).
    shift : Tensor
        Shift in (sub-)pixels/translation parameters, (B,2).
    padding : int, optional
        Width of the padding prior to convolution, by default 3.
    kernel_lobes : int, optional
        Number of lobes of the Lanczos interpolation kernel, by default 3.

    Returns
    -------
    _type_
        _description_
    """

    (batch_size, channels, height, width) = x.shape

    # Because examples and channels are interleaved in dim 1.
    shift = shift.repeat(channels, 1)  # (B, C * 2)
    shift = shift.reshape(batch_size * channels, 2)  # (B * C, 2)
    x = x.view(1, batch_size * channels, height, width)

    # Reflection pre-padding.
    pad = torch.nn.ReflectionPad2d(padding)
    x = pad(x)

    # 1D shifting kernels.
    y_shift = shift[:, [0]]
    x_shift = shift[:, [1]]

    # Flip dimension of convolution and expand dims to (batch_size, channels, len(kernel_y), 1).
    kernel_y = (lanczos_kernel(y_shift, kernel_lobes=kernel_lobes).flip(1))[
        :, None, :, None
    ]
    kernel_x = (lanczos_kernel(x_shift, kernel_lobes=kernel_lobes).flip(1))[
        :, None, None, :
    ]

    # 1D-convolve image with kernels: shifts image on x- then y-axis.
    x = torch.conv1d(
        x,
        groups=kernel_y.shape[0],
        weight=kernel_y,
        padding=[kernel_y.shape[2] // 2, 0],  # "same" padding.
    )
    x = torch.conv1d(
        x,
        groups=kernel_x.shape[0],
        weight=kernel_x,
        padding=[0, kernel_x.shape[3] // 2],
    )

    # Remove padding.
    x = x[..., padding:-padding, padding:-padding]

    return x.view(batch_size, channels, height, width)


class Shift:
    """ Sub-pixel image shifter with Lanczos shifting and interpolation kernels. """

    def __init__(self, padding=5, kernel_lobes=3):
        """ Initialize Shift.

        Parameters
        ----------
        padding : int, optional
            Width of the padding prior to convolution, by default 5.
        kernel_lobes : int, optional
            Number of lobes of the Lanczos interpolation kernel, by default 3.
        """
        self.padding = padding
        self.kernel_lobes = kernel_lobes

    def __call__(self, x, shift):
        """ Shift an image by convolving it with a Lanczos kernel.

        Parameters
        ----------
        x : Tensor
            Image to be shifted, (batch_size, channels, height, width).
        shift : Tensor
            Shift in (sub-)pixels/translation parameters, (batch_size,2).

        Returns
        -------
        Tensor
            Shifted image, (batch_size, channels, height, width).
        """
        return lanczos_shift(
            x, shift.flip(-1), padding=self.padding, kernel_lobes=self.kernel_lobes
        )