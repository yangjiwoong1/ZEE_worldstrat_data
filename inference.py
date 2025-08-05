import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
from multiprocessing import Manager
import numpy as np
from natsort import natsorted

# src.datasets와 srcnn을 임포트
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.datasets import load_dataset_JIF
from src.transforms import NormalizeInverse
from src.datasources import JIF_S2_MEAN, JIF_S2_STD, S2_ALL_12BANDS
from srcnn import SRCNN

def visualize_results(model, dataloader, device, num_samples, aoi_info_df):
    """
    모델의 추론 결과를 시각화하는 함수.
    LR, 생성된 SR, 원본 HR 이미지를 나란히 보여줍니다.
    타일 이름은 전달받은 데이터프레임(aoi_info_df)에서 가져옵니다.

    Warning: 타일 이름이 특정 조건에만 맞으며 정확하지 않을 수도 있습니다.
    """
    model.eval()

    # --- 역정규화 및 시각화용 정규화 함수 준비 ---
    lr_bands_index = np.array(S2_ALL_12BANDS["true_color"]) - 1
    lr_mean = JIF_S2_MEAN[lr_bands_index]
    lr_std = JIF_S2_STD[lr_bands_index]
    inv_normalize_lr = NormalizeInverse(mean=lr_mean, std=lr_std)

    def normalize_for_display(img):
        img = img - np.min(img)
        img = img / (np.max(img) + 1e-8)
        return np.clip(img, 0, 1)

    with torch.no_grad():
        # 파일명을 같이 출력하기 위해 인덱스로 접근
        for i, batch_dict in enumerate(dataloader):
            if i >= num_samples:
                break

            # i번째 데이터는 aoi_info_df의 i번째 행에 해당
            tile_name = aoi_info_df.iloc[i]['tile']

            lr_image_batch = batch_dict['lr'].to(device)
            hr_image = batch_dict['hr'][0] # 배치 차원 제거

            # 추론 후 배치 차원을 제거하고, 시각화를 위해 cpu 메모리로 옮김
            sr_image_tensor = model(lr_image_batch)[0].cpu()

            # 시각화를 위한 이미지 처리 (자세한 내용은 notebooks.dataloader_test.ipynb 참고)
            lr_denormalized = inv_normalize_lr(lr_image_batch[0].cpu())
            lr_display = normalize_for_display(lr_denormalized.numpy().transpose(1, 2, 0))

            sr_display = normalize_for_display(sr_image_tensor.numpy().transpose(1, 2, 0))
            hr_display = normalize_for_display(hr_image.numpy().transpose(1, 2, 0))

            # 이미지 시각화
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))

            axes[0].imshow(lr_display)
            axes[0].set_title('Low-Resolution (Input)')
            axes[0].axis('off')

            axes[1].imshow(sr_display)
            axes[1].set_title('Super-Resolution (Output)')
            axes[1].axis('off')

            axes[2].imshow(hr_display)
            axes[2].set_title('High-Resolution (Ground Truth)')
            axes[2].axis('off')

            plt.suptitle(f'Sample #{i+1} - Tile: {tile_name}', fontsize=16)
            plt.show()

def main():
    model_path = os.path.join("model", "srcnn_model_10epochs_4upscale.pth")
    # 인자 파싱
    parser = argparse.ArgumentParser(description='SRCNN Inference Script')
    parser.add_argument('--dataset', type=str, required=True, choices=['train', 'val', 'test'],
                        help='Dataset to use for inference (train, val, test)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--model_path', type=str, default=model_path,
                        help='Path to the trained SRCNN model file')
    parser.add_argument('--upscale_factor', type=int, default=4,
                        help='Upscale factor for the model')
    args = parser.parse_args()

    UPSCALE_FACTOR = args.upscale_factor
    INPUT_SIZE = (160, 160)
    OUTPUT_SIZE = (INPUT_SIZE[0] * UPSCALE_FACTOR, INPUT_SIZE[1] * UPSCALE_FACTOR)
    CHIP_SIZE = (160, 160)
    DATASET_ROOT = "dataset/"
    AOI_CSV_PATH = os.path.join(DATASET_ROOT, 'train_val_test_split.csv')

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 모델 로딩
    print(f"Loading model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    model = SRCNN(upscale_factor=UPSCALE_FACTOR).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model loaded.")

    # 데이터 로딩
    print(f"Loading {args.dataset} dataset...")
    if not os.path.exists(AOI_CSV_PATH):
        print(f"Error: AOI csv file not found at {AOI_CSV_PATH}")
        return

    list_of_aois = pd.read_csv(AOI_CSV_PATH)
    multiprocessing_manager = Manager()

    config = dict(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        chip_size=CHIP_SIZE,
        root=DATASET_ROOT,
        list_of_aois=list_of_aois,
        lr_bands_to_use="true_color",
        batch_size=1, # 시각화 비교를 위해 배치 사이즈를 1로 설정
        shuffle=False, # 추론 환경에선 셔플 x 
        randomly_rotate_and_flip_images=False,
        num_workers=0,
        use_single_frame_sr=True,
        multiprocessing_manager=multiprocessing_manager
    )

    dataloaders = load_dataset_JIF(**config)
    dataloader = dataloaders[args.dataset]
    print("Dataset loaded.")

    # 시각화용 AOI 이름 리스트를 만들 때, Dataset 내부 로직과 동일하게 처리
    aoi_info_for_vis = list_of_aois[list_of_aois['split'] == args.dataset].copy()
    sorted_tiles = natsorted(aoi_info_for_vis['tile'].tolist())
    aoi_info_for_vis['tile'] = pd.Categorical(aoi_info_for_vis['tile'], categories=sorted_tiles, ordered=True)
    aoi_info_for_vis.sort_values(by='tile', inplace=True, ignore_index=True)

    visualize_results(model, dataloader, device, args.num_samples, aoi_info_for_vis)

if __name__=='__main__':
    try:
        main()
    except SystemExit:
        print("\nInference script requires command-line arguments. Example:")
        print("python inference.py --dataset val --num_samples 3")