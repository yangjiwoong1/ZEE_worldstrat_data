#!/usr/bin/env python

import os
import json
import natsort
import numpy as np
import re
import warnings
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
import tifffile
import torch
from tqdm.auto import tqdm
from glob import glob
from typing import Callable
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from multiprocessing import Manager
from torchvision.transforms import Compose, Resize, InterpolationMode, Normalize, Lambda
import pandas as pd
from pathlib import Path
from PIL import Image
from src.datasources import (
    S2_ALL_12BANDS,
    SPOT_RGB_BANDS,
    JIF_S2_MEAN,
    JIF_S2_STD,
    S2_ALL_BANDS,
    SPOT_MAX_EXPECTED_VALUE_8_BIT,
    SPOT_MAX_EXPECTED_VALUE_12_BIT,
    ROOT_JIF_DATA_TRAIN,
    METADATA_PATH,
)
from src.transforms import (
    CropDict,
    RandomRotateFlipDict,
)

class SatelliteDataset(Dataset):
    """
    lr, lrc, hr (단일) 데이터셋 클래스
    """
    def __init__(
        self,
        root: str,
        number_of_revisits: int=None,
        subdir: str="",
        file_postfix: str=None,
        transform: Compose=None,
        bands_to_read: list[int]=None,
        file_reader: Callable=None,
        use_cache: bool=True,
        use_tifffile_as_reader: bool=False,
        list_of_aois: list[str]=None,
        multiprocessing_manager=None,
        load_metadata=False,
    ):
        """
        Params:
            root (str):
                - AOI별 하위 디렉토리 또는 AOI revisit별 파일을 포함하는 디렉토리 경로
                - 각 하위 디렉토리는 해당 AOI의 revisit을 포함

            subdir (str, optional):
                - revisit이 있는 루트 내의 하위 디렉토리, 기본값 '' (하위 디렉토리 없음)

            file_postfix (str, optional):
                - revisit 파일의 접미사, 기본값 'tif*'
                - 와일드카드 허용

            number_of_revisits (int, optional):
                - 각 AOI에 대해 가져올 revisit 수, 기본값 None (모든 revisit)

            transform (torchvision.transform, optional):
                - 각 revisit에 적용되는 변환, 기본값 Compose([]) (변환 없음)

            bands_to_read (list, optional):
                - 읽을 밴드 ID 목록, 기본값 None
                - GDAL 규칙에 따라 밴드 ID는 1부터 시작

            file_reader (callable, optional):
                - 파일을 읽는 데 사용되는 호출 가능한 함수, 기본값 SatelliteDataset._default_file_reader()

            use_cache (bool, optional):
                - 가져온 이미지를 메모리에 캐시, 기본값 True
                - 많은 RAM이 필요할 수 있지만 학습 속도를 크게 향상시킴
                - 디스크 I/O, 정규화, 리사이즈 -> 처음 한번만 수행하고 캐시에 저장

            use_tifffile_as_reader (bool, optional):
                - 이미지 로드에 tifffile 라이브러리 사용 여부, 기본값 False
                - False인 경우 rasterio 라이브러리가 사용됨

            list_of_aois (list, optional):
                - 루트에서 로드해야 하는 AOI list, 기본값 None

            multiprocessing_manager (multiprocessing.Manager, optional):
                - 여러 DataLoader 작업자가 캐시에 접근할 때 CPU 메모리 누수를 방지하는 데 사용(공유 캐시에 안전하게 접근), 기본값 None
                - Python의 참조 카운트로 인한 copy-on-write 동작을 방지하기 위해 multiprocessing.Manager 데이터 구조가 사용됨
                - Reference: https://github.com/pytorch/pytorch/issues/13246
        """

        self.check_root_is_not_empty(root)
        self.root = root
        self.subdir = subdir
        self.file_postfix = file_postfix if file_postfix is not None else "tif*"
        self.file_reader = file_reader or self._default_file_reader()
        self.transform = transform or Compose([])
        self.bands_to_read = bands_to_read
        self.number_of_revisits = number_of_revisits
        self.multiprocessing_manager = multiprocessing_manager
        self.paths = self.load_and_sort_aoi_paths(root, list_of_aois)
        self.use_cache = use_cache
        self.cache = self.multiprocessing_manager.dict()
        self.use_tifffile_as_reader = use_tifffile_as_reader
        self.load_metadata = load_metadata
        if self.load_metadata:
            self.metadata = pd.read_csv(METADATA_PATH, index_col=0)

    @staticmethod
    def check_root_is_not_empty(root):

        assert len(glob(root)) > 0, f"No files from {root}."

    def __len__(self):
        """
        AoI 개수 반환

        Return:
            AoI 개수
        """
        return len(self.paths)

    def __getitem__(self, item: int):
        """
        AOI 인덱스에 해당하는 이미지 반환
        - 이미지가 이전에 가져왔다면 캐시에서 반환
        - 그렇지 않다면 디스크에서 가져와서 변환(첫 에포크 로딩 시 느리지만 이후로는 캐싱 사용)

        Params:
            item (int): AOI 인덱스

        Return:
            Tensor: 텐서로 변환된 이미지
        """
        if item not in self.cache:
            path = self.paths[item]
            if os.path.isdir(path):
                # 경로가 디렉토리인 경우
                # print(f"Loading revisits from folder: {path}")
                return self.load_revisits_from_folder(path, item)
            else:
                # 경로가 파일인 경우
                # print(f"Loading revisits from file: {path}")
                return self.file_reader(path)
        else:
            # 이미 캐시에 있는 경우 캐시에서 반환
            return self.cache[item]

    def _default_file_reader(self):
        """
        기본 파일 읽기 함수 반환
        - self.use_tifffile_as_reader가 True인 경우 tifffile 라이브러리 사용
        - 파일이 PNG인 경우 PIL 라이브러리 사용
        - 그 외의 경우 rasterio 라이브러리 사용

        Return:
            Callable: 기본 파일 읽기 함수
        """

        def _reader(path: str):
            if self.load_metadata:
                aoi = Path(path).parent.stem
                return self.metadata.loc[aoi].head(1).to_dict("records")
            if path.endswith(".png"):
                return self.read_png(path)
            if self.use_tifffile_as_reader:
                return self.read_tiff_with_tifffile(path, self.bands_to_read)
            else:
                return self.read_tiff_with_rasterio(path, self.bands_to_read)

        return _reader

    @staticmethod
    def read_tiff_with_rasterio(path, bands_to_read=None):
        """
        TIFF 파일을 rasterio 라이브러리로 읽는 함수

        Params:
            path (str): TIFF 파일 경로
            bands_to_read (list, optional): 읽을 밴드 ID 목록, 기본값 None

        Return:
            np.ndarray: 이미지를 numpy 배열로 반환
        """
        return rasterio.open(path).read(indexes=bands_to_read, window=None)

    @staticmethod
    def read_tiff_with_tifffile(path, bands_to_read=None):
        """
        TIFF 파일을 tifffile 라이브러리로 읽는 함수

        Params:
            path (str): TIFF 파일 경로
            bands_to_read (list, optional): 읽을 밴드 ID 목록, 기본값 None

        Return:
            np.ndarray: 이미지를 numpy 배열로 반환
        """
        x = tifffile.imread(path)
        if bands_to_read is not None:
            x = x[..., bands_to_read]
        return x

    @staticmethod
    def read_png(path):
        """
        PNG 파일을 PIL 라이브러리로 읽는 함수

        Params:
            path (str): PNG 파일 경로

        Return:
            np.ndarray: 이미지를 numpy 배열로 반환
        """
        image = np.array(Image.open(path)).astype(np.float32)
        if len(image.shape) == 2:  # 2D/grayscale
            return image[None, ...]
        else:
            return image

    def load_and_sort_aoi_paths(self, root, list_of_aois=None):
        """
        AOI 경로 로드 및 자연 정렬을 수행하는 함수

        Params:
            list_of_aois (list, optional):
                - 루트에서 로드해야 하는 AOI list, 기본값 None
            root (str): 루트 디렉토리 경로

        Return:
            multiprocessing.Manager.list: 자연 정렬된 AOI 경로 목록
        """
        if list_of_aois is not None: # 명시적으로 지정된 AOI 목록이 있는 경우
            return self.multiprocessing_manager.list(
                natsort.natsorted([self.root.replace("*", aoi) for aoi in list_of_aois])
            )
        else: # 명시적으로 지정된 AOI 목록이 없는 경우 모든 AOI 로드
            return self.multiprocessing_manager.list(natsort.natsorted(glob(root)))

    def load_revisits_from_folder(self, path, item):
        """
        AOI 디렉토리에서 여러 revisit을 로드하는 함수

        Params:
            path (str): AOI 디렉토리 경로
            item (int): AOI 인덱스

        Return:
            Tensor: 여러 revisit을 텐서로 반환
        """
        # AOI 디렉토리에 여러 revisit이 있으므로, AOI-specific dataset으로 읽어옴
        aoi_dataset = self.generate_dataset_for_aoi_folder(path) # SatelliteDataset 객체 생성
        number_of_revisits = self.determine_the_number_of_revisits_to_return(aoi_dataset) # 로드할 revisit 수 결정(number_of_revisits 또는 디렉토리에 있는 revisit 수 중 최소값)

        if number_of_revisits > 0:
            x = self.load_revisits_from_aoi_dataset(number_of_revisits, aoi_dataset)
        else:
            # 여러 revisit이 없거나 number_of_revisits가 0인 경우
            return None
        x = self.transform(x) # 변환 적용
        self.cache_revisits(item, x)
        return x

    def generate_dataset_for_aoi_folder(self, path):
        """
        AOI 디렉토리에 대한 SatelliteDataset를 생성하는 함수

        Params:
            path (str): AOI 디렉토리 경로

        Return:
            SatelliteDataset: AOI 디렉토리에 대한 SatelliteDataset
        """
        aoi_dataset = SatelliteDataset(
            root=os.path.join(path, self.subdir, f"*{self.file_postfix}"),
            transform=self.transform,  # WARNING: random transforms vary among revisits!
            file_reader=self.file_reader,
            use_cache=False,
            multiprocessing_manager=self.multiprocessing_manager,
        )
        return aoi_dataset

    def determine_the_number_of_revisits_to_return(self, aoi_dataset):  
        """
        AOI 디렉토리에 있는 revisit 수를 결정하는 함수

        Params:
            aoi_dataset (SatelliteDataset): AOI 디렉토리에 대한 SatelliteDataset

        Return:
            int: 로드할 revisit 수
        """
        if self.number_of_revisits:
            # number_of_revisits가 지정된 경우, 지정된 수 또는 디렉토리에 있는 revisit 수 중 최소값 반환
            number_of_revisits = min(self.number_of_revisits, len(aoi_dataset))
        else:
            # number_of_revisits가 지정되지 않은 경우, 모든 revisit 반환
            number_of_revisits = len(aoi_dataset)
        return number_of_revisits

    def load_revisits_from_aoi_dataset(self, number_of_revisits, aoi_dataset):
        """
        AOI 디렉토리에 있는 revisit을 로드하는 함수

        Params:
            number_of_revisits (int): 로드할 revisit 수
            aoi_dataset (SatelliteDataset): AOI 디렉토리에 대한 SatelliteDataset

        Return:
            Tensor: 여러 revisit을 텐서로 반환
        """
        x = np.stack([aoi_dataset[revisit] for revisit in range(number_of_revisits)], axis=0)

        if self.use_tifffile_as_reader:
            if x.ndim == 3:  # If image is grayscale
                # Add empty dimension for channels
                x = x[..., None]
            number_of_revisits, height, width, channels = 0, 1, 2, 3
            # Convert from channel-last to channel-first
            x = x.transpose(number_of_revisits, channels, height, width)
        return x

    def cache_revisits(self, item, x):
        """
        revisit을 캐시에 저장하는 함수
        - load_revisits_from_folder에서 각 AOI 폴더에 대한 임시 aoi_dataset 객체를 생성하는 것에 대한 비효율성 완화
        - 전처리(디스크 I/O, 정규화, 리사이즈)의 비효율 완화
        - 첫 에포크의 로딩은 느리지만 이후로는 캐싱 사용

        Params:
            item (int): AOI 인덱스
            x (Tensor): 저장할 revisit 텐서
        """
        if self.use_cache and self.file_postfix != "_pan.tiff":
            # pan-chromatic channels/images는 캐시에 저장하지 않음(RAM 절약)
            self.cache[item] = x

    def compute_median_std(self, name=None):
        """
        데이터셋의 각 밴드/채널에 대한 중앙값과 중앙값-표준편차를 계산하는 함수
        Note: 표준편차는 중앙값을 기준으로 계산되며, 일반적인 표준편차와는 다름!!

        Params:
            name (str, optional): 데이터셋 이름, 기본값 None

        Return:
            torch.Tensor, torch.Tensor: 각 밴드/채널에 대한 중앙값과 중앙값-표준편차
        """

        progress_bar_description = f" for {name}" if name is not None else ""
        number_of_channels = self[0].shape[1]
        number_of_revisits, channels, height, width = 0, 1, 2, 3

        channels_over_all_revisits = torch.cat(
            [
                x.to(float)
                # Permute to channel-first
                .permute(dims=(channels, number_of_revisits, height, width))
                # Flatten to [number_of_channels, number_of_channels * height * width]
                .reshape(number_of_channels, -1)
                # Progress bar
                for x in tqdm(
                    self, desc=f"Calculating median and std{progress_bar_description}"
                )
            ],
            # Concatenate along the channel axis
            dim=1,
        )

        # Compute the median by reducing along the channel axis
        median = channels_over_all_revisits.median(dim=1, keepdims=True).values
        # Compute the median-standard deviation by subtracting the median from the channel-wise median-standard deviation
        std = ((channels_over_all_revisits - median) ** 2).mean(dim=1).sqrt()
        return median.squeeze(), std


class SingleFrameSatelliteDataset(SatelliteDataset):
    """
    SatelliteDataset을 상속받아, 여러 LR revisit 중 metadata에 정의된 cloud_cover가 가장 낮은
    단일 프레임만 선택하여 로드하는 데이터셋 클래스
    TODO: 메타데이터의 cloud_cover는 AOI가 아닌 전체 영역에 대한 수치이므로 최적화 필요
    """
    def load_revisits_from_folder(self, path, item):
        """
        기존의 여러 revisit을 스택하는 대신, metadata를 읽어
        cloud_cover가 가장 낮은 단일 이미지를 선택 및 로드하는 함수

        Params:
            path (str): AOI 디렉토리 경로
            item (int): AOI 인덱스

        Return:
            Tensor: 선택된 단일 revisit을 텐서로 반환
        """
        # AOI 폴더 내의 모든 이미지와 메타데이터 파일 경로를 탐색
        revisit_paths = natsort.natsorted(glob(os.path.join(path, self.subdir, f"*{self.file_postfix}")))
        metadata_paths = {os.path.basename(p).split('.')[0]: p for p in glob(os.path.join(path, '*.metadata'))}

        if not revisit_paths:
            return None

        # 메타데이터를 읽어 cloud_cover 값과 이미지 경로를 매핑
        revisit_info = []
        for revisit_path in revisit_paths:
            base_name = os.path.basename(revisit_path)
            # 정규표현식을 사용하여 파일 이름에서 revisit_id(e.g., 'Landcover-778183-1') 추출
            match = re.match(r'(.*-\d+)', base_name)
            if not match:
                print(f"[WARNING] Could not determine revisit_id for: {base_name}")
                continue
            revisit_id = match.group(1) # 

            cloud_cover = float('inf')  # 메타데이터가 없거나 키가 없으면 무한대로 설정

            if revisit_id in metadata_paths:
                metadata_path = metadata_paths[revisit_id]
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        # "cloud_cover" 키가 없는 경우에 대한 처리
                        if "cloud_cover" in metadata:
                            cloud_cover = metadata["cloud_cover"]
                        else:
                            print(f"[WARNING] 'cloud_cover' key not found in metadata for: {metadata_path}")
                except json.JSONDecodeError:
                    print(f"[ERROR] JSONDecodeError in metadata file: {metadata_path}")
                except FileNotFoundError:
                    print(f"[ERROR] Metadata file not found: {metadata_path}")
                except Exception as e:
                    print(f"[ERROR] Unexpected error while reading metadata for {metadata_path}: {e}")
            else:
                print(f"[WARNING] Metadata file not found for revisit_id: {revisit_id}")

            revisit_info.append({'path': revisit_path, 'cloud_cover': cloud_cover})


        # Cloud cover를 기준으로 정렬(가장 낮은 값이 맨 앞으로)
        # 메타데이터가 없는 경우(inf)는 뒤로 밀려남
        sorted_revisits = sorted(revisit_info, key=lambda x: x['cloud_cover'])

        # 가장 cloud_cover가 낮은 이미지 경로를 선택
        # 만약 모든 파일에 메타데이터가 없다면, 정렬된 리스트의 첫 번째 이미지를 사용
        best_revisit_path = sorted_revisits[0]['path']

        # 선택된 단일 이미지를 로드
        # file_reader는 (c, h, w) 형태의 numpy array를 반환 -> Tensor로 변환 필요
        x = self.file_reader(best_revisit_path)

        # 원본 `load_revisits_from_aoi_dataset`의 후처리 로직 일부를 가져와 처리
        if self.use_tifffile_as_reader:
            if x.ndim == 2:  # Grayscale
                x = x[..., None] # (h, w) -> (h, w, 1)
            # (h, w, c) -> (c, h, w)
            x = x.transpose(2, 0, 1)

        x = self.transform(torch.as_tensor(x))  # transform 이전에 텐서로 변환
        self.cache_revisits(item, x)

        return x

class TransformDataset(Dataset):
    """
    PyTorch Dataset 클래스를 사용하여 이미지를 가져올 때 변환을 적용하는 클래스
    Reference: https://gist.github.com/alkalait/c99213c164df691b5e37cd96d5ab9ab2#file-sn7dataset-py-L278
    """
    def __init__(self, dataset: Dataset, transform: Callable) -> None:
        """
        PyTorch Dataset 클래스를 사용하여 이미지를 가져올 때 변환을 적용하는 클래스

        Params:
            dataset (torch.Dataset): 변환을 적용할 데이터셋
            transform (Callable): 적용할 변환
        """
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item: int):
        """
        데이터셋에서 이미지를 가져오고 변환을 적용하는 함수

        Params:
            item (int): 가져올 이미지의 인덱스

        Return:
            Tensor: 변환이 적용된 이미지
        """
        item = self.dataset[item]
        return self.transform(item)

    def __len__(self):
        return len(self.dataset)


class DictDataset(Dataset):
    """
    PyTorch Dataset 클래스를 사용하여 데이터셋 딕셔너리를 랩핑하는 클래스
    - get 메서드에 전달된 인덱스는 단일 AOI의 이미지 또는 크롭(크롭 적용 시)을 참조
    - get 메서드는 각 데이터셋에 대한 이미지 인덱스의 이미지 딕셔너리를 반환
    """

    def __init__(self, **dictionary_of_datasets):
        """

        Params:
            dictionary_of_datasets (dict): 랩핑할 torch 데이터셋 딕셔너리
        """
        self.datasets = {
            dataset_name: dataset
            for dataset_name, dataset in dictionary_of_datasets.items()
            if isinstance(dataset, Dataset)
        }

    def __getitem__(self, item):
        """
        데이터셋에서 이미지를 가져오는 함수

        Params:
            item (int): 가져올 이미지의 인덱스

        Return:
            dict: 각 데이터셋에서 가져온 이미지(dataset_name: item)
        """
        data = {name: dataset[item] for name, dataset in self.datasets.items()}

        # hr의 revisit 차원 squeeze
        if 'hr' in data and data['hr'] is not None:
            # 차원이 4개(R, C, H, W)이고, revisit 차원의 크기가 1인지 확인
            if data['hr'].ndim == 4 and data['hr'].shape[0] == 1:
                data['hr'] = data['hr'].squeeze(0)

        return data

    def __len__(self):
        """
        랩핑된 데이터셋 딕셔너리에서 가장 작은 데이터셋의 길이를 반환하는 함수

        Return:
            int: 랩핑된 데이터셋 딕셔너리에서 가장 작은 데이터셋의 길이
        """
        return min(len(dictionary) for dictionary in self.datasets.values())


def load_dataset_JIF(**kws) -> dict[str, DataLoader]:
    """
    Params:
        input_size: LR 리사이즈 크기
        output_size: HR 리사이즈 크기
        chip_size: LR 칩 크기(default: input_size)
        chip_stride: LR 칩 스트라이드(default: chip_size)
        revisits: revisit 수(default: 8)
        batch_size: 배치 크기(default: 1)
        batch_size_test: 테스트 배치 크기(default: 1)
        normalize_lr: LR 정규화 여부
        shuffle: 데이터셋 셔플 여부(default: True)
        interpolation: LR 리사이즈 방법
        lr_bands_to_use: LR 밴드 대역 선택(default: all(12bands) | true_color(3bands)) 
        radiometry_depth: HR 데이터 비트 깊이(default: 12bit)
        randomly_rotate_and_flip_images: 데이터 회전 및 뒤집기 여부(default: False)
        list_of_aois: AOI 리스트(default: None)
        root: 데이터셋 루트 경로(default: dataset/)
        data_split_seed: 데이터셋 셔플 시드(default: 42)
        compute_median_std: 데이터셋 중앙값/표준편차 계산 여부(default: False) - 정규화를 위한 통계치
        subset_train: 트레이닝 데이터셋 비율(default: None | 0.0 ~ 1.0)
        use_single_frame_sr: LR 채널에 대해 단일 프레임 SR 사용 여부(default: True)

    Return:
        dict: 데이터로더(test, train, val)
        (e.g., {'test': DataLoader, 'train': DataLoader, 'val': DataLoader})
    """

    kws.setdefault("input_size", (160, 160))
    kws.setdefault("output_size", (1054, 1054))
    kws.setdefault("chip_size", kws["input_size"])
    kws.setdefault("chip_stride", kws["chip_size"])
    kws.setdefault("batch_size", 1)
    kws.setdefault("batch_size_test", 1)
    kws.setdefault("revisits", 8)

    kws.setdefault("normalize_lr", True)
    kws.setdefault("shuffle", True)
    kws.setdefault("interpolation", InterpolationMode.BILINEAR)

    kws.setdefault("lr_bands_to_use", "all")
    kws.setdefault("radiometry_depth", 12)
    kws.setdefault("randomly_rotate_and_flip_images", False)
    kws.setdefault("list_of_aois", None)

    kws.setdefault("root", "dataset/")
    kws.setdefault("train_split", None)
    kws.setdefault("val_split", None)
    kws.setdefault("test_split", None)
    kws.setdefault("data_split_seed", 42)
    kws.setdefault("compute_median_std", False)
    kws.setdefault("subset_train", None)
    kws.setdefault("use_single_frame_sr", True)
    kws["number_of_revisits"] = kws["revisits"]

    kws["root"] = set_subfolders_for_roots_JIF(kws["root"], kws["radiometry_depth"])
    hr_postfix = "_ps_8bit.tiff" if kws["radiometry_depth"] == 8 else "_ps.tiff"
    lr_bands = S2_ALL_12BANDS["true_color"] if kws["lr_bands_to_use"] == "true_color" else S2_ALL_BANDS
    transforms = make_transforms_JIF(**kws)

    return make_dataloaders(
        subdir={"lr": "", "lrc": "", "hr": ""},#, "metadata": ""},
        bands_to_read={
            "lr": lr_bands,
            "lrc": None,
            "hr": SPOT_RGB_BANDS,
            #"hr_pan": None,
            #"metadata": None,
        },
        transforms=transforms,
        file_postfix={
            "lr": "-L2A_data.tiff",
            "lrc": "-CLM.tiff",
            "hr": hr_postfix,
            #"hr_pan": hr_pan_postfix,
            #"metadata": hr_postfix,
        },
        **kws,
    )


def make_dataloaders(
    root: dict[str, str],
    subdir: dict[str, str],
    input_size: tuple[int, int],
    chip_size: tuple[int, int],
    chip_stride: int,
    bands_to_read: dict[str, list[int]],
    transforms: dict[str, Compose],
    shuffle: bool = True,
    data_split_seed: int=42,
    number_of_revisits:int=None,
    file_postfix: dict[str, str]=None,
    use_tifffile_as_reader=None,
    subset_train=None,
    randomly_rotate_and_flip_images=True,
    compute_median_std=False,
    list_of_aois:pd.DataFrame=None,
    use_single_frame_sr=True,
    **kws,
) -> dict[str, DataLoader]:
    """
    데이터로더를 생성하는 함수
    """
    number_of_revisits, use_tifffile_as_reader= set_default_argument_values(number_of_revisits, use_tifffile_as_reader, transforms.keys())

    multiprocessing_manager = Manager()
    satellite_datasets_arguments = generate_satellite_dataset_arguments_from_kws(
        root,
        subdir,
        file_postfix,
        bands_to_read,
        transforms,
        use_tifffile_as_reader,
        number_of_revisits,
        multiprocessing_manager,
        list_of_aois,
    ) # dict[dict[str, any]] (e.g., {'lr': {'root': 'dataset/lr_dataset/*/L2A/', 'subdir': '', 'file_postfix': '-L2A_data.tiff', 'bands_to_read': [4, 3, 2], 'transform': <Compose>, 'use_tifffile_as_reader': True, 'number_of_revisits': 8, 'multiprocessing_manager': <multiprocessing.managers.SyncManager object at 0x7f5010000000>, 'list_of_aois': <DataFrame>})

    datasets = generate_satellite_datasets(
        satellite_datasets_arguments, list_of_aois, compute_median_std, subdir, use_single_frame_sr
    ) # dict[str, DictDataset] (e.g., {'train': <DictDataset>, 'val': <DictDataset>, 'test': <DictDataset>})

    if shuffle:
        datasets = shuffle_datasets(datasets, data_split_seed)
    datasets, number_of_chips = generate_chipped_and_augmented_datasets(
        datasets,
        chip_size,
        chip_stride,
        input_size,
        randomly_rotate_and_flip_images,

    ) # 변환을 적용한 ConcatDataset(DictDataset -> ConcatDataset)

    if type(datasets) is dict:
        dataset_train, dataset_val, dataset_test = (
            datasets["train"][0], # 'train': (train_dataset_object, train_chip_count)
            datasets["val"][0],
            datasets["test"][0],
        )

    if subset_train is not None:
        dataset_train = reduce_training_set(dataset_train, subset_train)

    test_dataloader, train_dataloader, val_dataloader = create_dataloaders_for_datasets(
        dataset_test, dataset_train, dataset_val, kws
    )

    print(f"Train set size: {len(dataset_train)}")
    print(f"Val set size: {len(dataset_val)}")
    print(f"Test set size: {len(dataset_test)}")

    return {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}


def set_default_argument_values(
    number_of_revisits,
    use_tifffile_as_reader,
    dataset_keys:list[str],
) -> tuple[dict[str, int], dict[str, bool]]:
    """
    Params:
        dataset_keys: 데이터셋 키(lr, lrc, hr)
        number_of_revisits: revisits 수
        use_tifffile_as_reader: tifffile 사용 여부

    Return:
        tuple[dict[str, int], dict[str, bool]]: 데이터셋 키(lr, lrc, hr)에 대한 딕셔너리
    """
    none_dictionary = {key: None for key in dataset_keys}

    if number_of_revisits is None:
        number_of_revisits = none_dictionary
    elif isinstance(number_of_revisits, int):
        number_of_revisits = {
            "lr": number_of_revisits,
            "lrc": number_of_revisits,
            "hr": 1,
            # "metadata": 1,
        }
    use_tifffile_as_reader = (
        none_dictionary if use_tifffile_as_reader is None else use_tifffile_as_reader
    )
    return number_of_revisits, use_tifffile_as_reader


def generate_satellite_dataset_arguments_from_kws(
    root,
    subdir,
    file_postfix,
    bands_to_read,
    transforms,
    use_tifffile_as_reader,
    number_of_revisits,
    multiprocessing_manager,
    list_of_aois,
) -> dict[str, dict]:

    return {
        dataset_name: dict(
            root=root[dataset_name],
            subdir=subdir[dataset_name],
            file_postfix=file_postfix[dataset_name],
            bands_to_read=bands_to_read[dataset_name],
            transform=transforms[dataset_name],
            use_tifffile_as_reader=use_tifffile_as_reader[dataset_name],
            number_of_revisits=number_of_revisits[dataset_name],
            multiprocessing_manager=multiprocessing_manager,
            list_of_aois=list_of_aois,
        )
        for dataset_name in subdir
    }


def generate_satellite_datasets(
    satellite_dataset_arguments: dict[dict[str, any]], 
    list_of_aois: pd.DataFrame,
    compute_median_std: bool,
    subdir: dict[str, str],
    use_single_frame_sr: bool,
) -> dict[str, DictDataset]: # (e.g., {'train': <DictDataset>, 'val': <DictDataset>, 'test': <DictDataset>})

    if isinstance(list_of_aois, pd.DataFrame) and "split" in list_of_aois.columns:

        assert {"train", "test", "val"} == set(
            list_of_aois["split"]
        ), "The list of AOIs needs to have a train/test/val split"

        return {
            split: generate_datasets_for_split(
                split,
                satellite_dataset_arguments,
                list_of_aois,
                compute_median_std,
                subdir,
                use_single_frame_sr,
            )
            for split in ["train", "val", "test"]
        }


def generate_datasets_for_split(
    split, satellite_datasets_arguments, list_of_aois, compute_median_std, subdir, use_single_frame_sr
):
    split_aois = list_of_aois[list_of_aois["split"] == split]
    datasets_arguments = satellite_datasets_arguments.copy()

    if pd.api.types.is_numeric_dtype(split_aois.index): # 정수형 인덱스인지 검사
        aoi_list = list(split_aois.iloc[:, 1])
    else:
        aoi_list = list(split_aois.index)

    for dataset in datasets_arguments:
        datasets_arguments[dataset]["list_of_aois"] = aoi_list # 각 데이터셋(lr, lrc, hr)에 대한 AOI 리스트 설정(데이터프레임이 아닌 리스트로 변환)

    return generate_satellite_datasets_without_pansharpening(
        datasets_arguments,
        compute_median_std,
        subdir,
        use_single_frame_sr,
    )


def generate_satellite_datasets_without_pansharpening(satellite_dataset_arguments, compute_median_std, subdir, 
                                                      use_single_frame_sr=False):
    """
    팬샤프닝 없이 데이터셋을 생성하는 함수(이미 펜샤프닝 된 데이터셋을 사용)

    Params:
        satellite_dataset_arguments (dict[str, dict]): 데이터셋 인수
        compute_median_std (bool): 데이터셋 중앙값/표준편차 계산 여부
        subdir (dict[str, str]): 데이터셋 하위 디렉토리
        use_single_frame_sr (bool): LR 채널에 대해 단일 프레임 SR 사용 여부

    Return:
        DictDataset: DictDataset 객체
    """
    datasets = {}
    for dataset_name, arguments in satellite_dataset_arguments.items():
        if use_single_frame_sr and dataset_name in ['lr', 'lrc']:
            print(f"Using SingleFrameSatelliteDataset for {dataset_name.upper()} channel.")
            datasets[dataset_name] = SingleFrameSatelliteDataset(**arguments)
        else:
            datasets[dataset_name] = SatelliteDataset(**arguments)

    if compute_median_std:
        compute_median_std_for_datasets(datasets, subdir)

    dataset_dict = DictDataset(
        **datasets
    )
    return dataset_dict


def compute_median_std_for_datasets(datasets, subdir):

    for dataset_name in subdir:
        dataset = datasets[dataset_name]
        print(f"{dataset_name}:{dataset.compute_median_std(name=dataset_name)}")


def shuffle_datasets(datasets, data_split_seed: int):
    """
    train, val, test 데이터셋 각각을 랜덤하게 섞는 함수
    """
    print(f"Shuffling the dataset splits using {data_split_seed}")

    if isinstance(datasets, dict):
        return {
            key: shuffle_datasets(value, data_split_seed)
            for key, value in datasets.items()
        }
    number_of_scenes = len(datasets)
    (datasets,) = random_split(
        datasets,
        [number_of_scenes,],
        generator=torch.Generator().manual_seed(data_split_seed),
    )
    return datasets


def generate_chipped_and_augmented_datasets(
    datasets,
    chip_size,
    chip_stride,
    input_size,
    randomly_rotate_and_flip_images,
) -> tuple[dict[str, DictDataset], None]:

    if isinstance(datasets, dict):
        return (
            {
                key: generate_chipped_and_augmented_datasets(
                    value,
                    chip_size,
                    chip_stride,
                    input_size,
                    randomly_rotate_and_flip_images,
                )
                for key, value in datasets.items()
            },
            None,
        )
    number_of_scenes = len(datasets)

    randomly_rotate_and_flip_images = generate_random_rotation_and_flip_transform(
        randomly_rotate_and_flip_images
    )
    # Concatenate chipped views of the scene-level dataset, with a sliding window.
    dataset, number_of_chips = apply_crop_and_rotations_to_datasets(
        datasets,
        input_size,
        chip_size,
        chip_stride,
        randomly_rotate_and_flip_images,
    )
    dataset = transpose_scenes_and_chips(dataset, number_of_chips, number_of_scenes) # 다양성을 위해 씬 수준 데이터셋을 칩 수준 데이터셋으로 변환
    return dataset, number_of_chips


def generate_random_rotation_and_flip_transform(randomly_rotate_and_flip_images):
    """
    랜덤하게 회전 및 뒤집기 변환을 생성하는 함수
    """
    if randomly_rotate_and_flip_images:
        randomly_rotate_and_flip_images = RandomRotateFlipDict(angles=[0, 90, 180, 270])
    else:
        randomly_rotate_and_flip_images = Compose([])
    return randomly_rotate_and_flip_images


def apply_crop_and_rotations_to_datasets(
    dataset_dict,
    input_size,
    chip_size,
    chip_stride,
    randomly_rotate_and_flip_images,
):
    """
    데이터셋을 크롭 및 회전 및 뒤집기 변환을 적용하는 함수

    Params:
        dataset_dict (DictDataset of SatelliteDataset): 크롭 및 회전할 데이터셋
        input_size (int): 입력 이미지 크기
        chip_size (int): 칩/패치 크기
        chip_stride (int): 칩/패치 스트라이드
        randomly_rotate_and_flip_images (bool): 이미지 랜덤 회전 및 뒤집기 여부

    Return:
        DictDataset of SatelliteDataset: 크롭 및 회전된 데이터셋
    """
    dataset_dict_grid = []
    input_height, input_width = input_size
    chip_height, chip_width = chip_size
    stride_height, stride_width = chip_stride

    # Make sure chip isn't larger than the input size
    assert chip_height <= input_height and chip_width <= input_height

    last_stride_step_x = input_width - chip_width + 1
    last_stride_step_y = input_height - chip_height + 1
    for stride_step_x in range(0, last_stride_step_x, stride_width):
        for stride_step_y in range(0, last_stride_step_y, stride_height):
            transform_dict = Compose(
                [
                    CropDict(
                        stride_step_x, stride_step_y, chip_width, chip_height, src="lr"
                    ),
                    randomly_rotate_and_flip_images,
                ]
            )
            dataset_dict_grid.append(
                TransformDataset(dataset_dict, transform=transform_dict)
            )
    dataset = torch.utils.data.ConcatDataset(dataset_dict_grid) # 데이터셋 여러 개를 합침(ConcatDataset)
    return dataset, len(dataset_dict_grid)


def transpose_scenes_and_chips(dataset, number_of_chips, number_of_scenes):
    """
    데이터셋 샘플을 재정렬하여 서로 다른 장면의 칩들이 교차되도록하는 함수

    데이터셋이 여러 개의 칩을 포함하는 장면들로 구성되어 있다고 가정
    이 함수는 인덱스를 재배열하여 다음과 같은 순서의 데이터셋을 생성:
    [chip0_scene0, chip0_scene1, ..., chip0_sceneN, chip1_scene0, chip1_scene1, ..., chip1_sceneN, ...]

    Params:
        dataset (torch.utils.data.Dataset): 장면 단위로 순차적으로 구성된 원본 데이터셋
        number_of_scenes (int): 데이터셋의 고유한 장면 수
        number_of_chips (int): 장면당 칩의 수

    Return:
        torch.utils.data.Subset: 샘플이 재정렬된 원본 데이터셋의 Subset
    """
    # Transpose scenes and chips
    # indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ..., number_of_scenes * number_of_chips]
    indices = Tensor(range(number_of_scenes * number_of_chips)).int()
    # indices = [0, number_of_scenes, 2*number_of_scenes, ..., 1+number_of_scenes, 2+2*number_of_scenes, ... ]
    transposed_indices = indices.reshape(number_of_chips, number_of_scenes).T.reshape(
        indices.numel()
    )
    dataset = torch.utils.data.Subset(dataset, transposed_indices)
    assert len(dataset) == number_of_scenes * number_of_chips
    return dataset


def reduce_training_set(dataset_train, subset_train):
    """
    훈련 데이터셋을 주어진 비율로 줄이는 함수

    Params:
        dataset_train (SatelliteDataset): 훈련 데이터셋
        subset_train (float): 훈련 데이터셋 사용 비율

    Return:
        SatelliteDataset: 줄여진 훈련 데이터셋
    """
    # Reduce the train set if needed
    if subset_train < 1:
        dataset_train = torch.utils.data.Subset(
            dataset_train, list(range(int(subset_train * len(dataset_train))))
        )
    return dataset_train


def create_dataloaders_for_datasets(dataset_test, dataset_train, dataset_val, kws):
    """
    데이터셋에서 PyTorch 데이터로더를 생성하는 함수

    Params:
        dataset_test (torch.utils.data.Dataset): 테스트 데이터셋
        dataset_train (torch.utils.data.Dataset): 훈련 데이터셋
        dataset_val (torch.utils.data.Dataset): 검증 데이터셋
        kws (dict): 데이터로더 생성에 사용될 키워드 인자

    Return:
        torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader: 테스트, 훈련 및 검증 데이터셋에 대한 데이터로더
    """
    batch_size, batch_size_test, number_of_workers = kws.get("batch_size", 1), kws.get("batch_size_test", 1), kws.get("num_workers", 1)
    train_dataloader = DataLoader(
        dataset_train,
        num_workers=number_of_workers,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        persistent_workers=True if number_of_workers > 0 else False,
    )
    # val_loader = DataLoader(dataset_val, num_workers=W, batch_size=len(dataset_val), pin_memory=True,
    val_dataloader = DataLoader(
        dataset_val,
        num_workers=number_of_workers,
        batch_size=batch_size,
        pin_memory=True,
        persistent_workers=True if number_of_workers > 0 else False,
    )
    # test_loader = DataLoader(dataset_test, num_workers=W, batch_size=len(dataset_test), pin_memory=True)
    test_dataloader = DataLoader(
        dataset_test, num_workers=number_of_workers, batch_size=batch_size_test, pin_memory=True
    )
    return test_dataloader, train_dataloader, val_dataloader


def make_transforms_JIF(
    input_size=(160, 160),
    output_size=(1054, 1054),
    interpolation=InterpolationMode.BICUBIC,
    normalize_lr=True,
    radiometry_depth=12,
    lr_bands_to_use="all",
    **kws,
) -> dict[str, Compose]:
    """
    lr, lrc, hr 데이터셋에 대한 변환 함수를 생성하는 함수

    Params:
        input_size: LR 리사이즈 크기
        output_size: HR 리사이즈 크기
        interpolation: LR 리사이즈 방법
        normalize_lr: LR 정규화 여부
        radiometry_depth: HR 데이터 비트 깊이(default: 12bit)
        lr_bands_to_use: LR 밴드 대역 선택(default: all(12bands) | true_color(3bands)) 

    return:
        dict: 변환 함수(lr, lrc, hr)
    """

    maximum_expected_hr_value = SPOT_MAX_EXPECTED_VALUE_12_BIT if radiometry_depth == 12 else SPOT_MAX_EXPECTED_VALUE_8_BIT
    lr_bands_index = np.array(S2_ALL_12BANDS["true_color"]) - 1 if lr_bands_to_use == "true_color" else np.array(S2_ALL_BANDS) - 1
    lr_mean = JIF_S2_MEAN[lr_bands_index]
    lr_std = JIF_S2_STD[lr_bands_index]

    if normalize_lr:
        normalize = Normalize(mean=lr_mean, std=lr_std)
    else:
        normalize = Compose([])

    transforms = {}
    transforms["lr"] = Compose(
        [
            Lambda(lambda lr_revisit: torch.as_tensor(lr_revisit)),
            normalize,
            Resize(size=input_size, interpolation=interpolation, antialias=True),
        ]
    )

    transforms["lrc"] = Compose(
        [
            Lambda(
                lambda lr_cloud_mask: torch.as_tensor(lr_cloud_mask)
            ),
            # Categorical
            Resize(size=input_size, interpolation=InterpolationMode.NEAREST),
        ]
    )

    transforms["hr"] = Compose(
        [
            Lambda(
                lambda hr_revisit: torch.as_tensor(hr_revisit.astype(np.int32))
                / maximum_expected_hr_value
            ),
            Resize(size=output_size, interpolation=interpolation, antialias=True),
            Lambda(lambda high_res_revisit: high_res_revisit.clamp(min=0, max=1)),
        ]
    )

    return transforms


def set_subfolders_for_roots_JIF(root: str, radiometry_depth: int) -> dict[str, str]:
    """
    Params:
        root: 데이터셋 루트 경로
        radiometry_depth: HR 데이터 비트 깊이(default: 12bit)

    Return:
        dict: 데이터셋 디렉토리 경로(lr, lrc, hr)
    """
    if radiometry_depth == 8:
        return {
            "lr": os.path.join(root, "lr_dataset", "*", "L2A", ""),
            "lrc": os.path.join(root, "lr_dataset", "*", "L2A", ""),
            "hr": os.path.join(root, "hr_dataset", "8bit", "*", ""),
            #"hr_pan": os.path.join(root, "hr_dataset", "8bit", "*", ""),
            #"metadata": os.path.join(root, "hr_dataset", "8bit", "*", ""),
        }
    else:
        return {
            "lr": os.path.join(root, "lr_dataset", "*", "L2A", ""), # 디렉토리만 매칭
            "lrc": os.path.join(root, "lr_dataset", "*", "L2A", ""),
            "hr": os.path.join(root, "hr_dataset", "*", ""),
            #"hr_pan": os.path.join(root, "hr_dataset", "*", ""),
            #"metadata": os.path.join(root, "hr_dataset",  "*", ""),
        }

if __name__ == "__main__":
    """ Median and std of dataset bands. """
    dataset_lr = SatelliteDataset(
        root=ROOT_JIF_DATA_TRAIN, subdir="S2", number_of_revisits=10
    )
    dataset_hr = SatelliteDataset(
        root=ROOT_JIF_DATA_TRAIN, subdir="images", number_of_revisits=1
    )
    print(f"S2 median / std: {dataset_lr.compute_median_std()}")
    print(f"PlanetScope median / std: {dataset_hr.compute_median_std()}")
