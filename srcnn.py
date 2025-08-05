import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from multiprocessing import Manager
import os
from tqdm import tqdm
from src.datasets import load_dataset_JIF

class SRCNN(nn.Module):

    def __init__(self, upscale_factor=4, in_channels=3, out_channels=3):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bicubic', align_corners=False)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=5, padding=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_upscaled = self.upsample(x)
        out = self.relu(self.conv1(x_upscaled))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out

# 메인 학습 스크립트
def main():
    # --- 파라미터 설정 --- #
    UPSCALE_FACTOR = 6 # x N Super Resolution
    INPUT_SIZE = (160, 160)
    OUTPUT_SIZE = (INPUT_SIZE[0] * UPSCALE_FACTOR, INPUT_SIZE[1] * UPSCALE_FACTOR)
    CHIP_SIZE = (160, 160)
    BATCH_SIZE = 6
    NUM_EPOCHS = 2
    LEARNING_RATE = 1e-4
    DATASET_ROOT = "dataset"
    AOI_CSV_PATH = os.path.join(DATASET_ROOT, 'train_val_test_split.csv')

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 데이터 로딩
    print("Loading dataset...")
    if not os.path.exists(AOI_CSV_PATH):
        print(f"Error: AOI csv file not found at {AOI_CSV_PATH}")
        return

    list_of_aois = pd.read_csv(AOI_CSV_PATH)
    multiprocessing_manager = Manager()

    # JIF 데이터셋 로더를 위한 파라미터
    config = dict(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        chip_size=CHIP_SIZE,
        batch_size=BATCH_SIZE,
        root=DATASET_ROOT,
        list_of_aois=list_of_aois,
        lr_bands_to_use="true_color", # RGB
        randomly_rotate_and_flip_images=True,
        num_workers=0,
        use_single_frame_sr=True,  # 단일 프레임 모드
        multiprocessing_manager=multiprocessing_manager
    )

    # 데이터로더 생성
    dataloaders = load_dataset_JIF(**config)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    print("Dataset loaded.")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    print("Initializing model...")
    model = SRCNN(upscale_factor=UPSCALE_FACTOR).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 학습 루프
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)

        for batch_dict in progress_bar:
            lr_images = batch_dict['lr'].to(device)
            hr_images = batch_dict['hr'].to(device)

            optimizer.zero_grad()
            sr_images = model(lr_images)

            loss = criterion(sr_images, hr_images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Training Loss: {epoch_loss:.4f}")

        # 검증
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_dict in val_loader:
                lr_images = batch_dict['lr'].to(device)
                hr_images = batch_dict['hr'].to(device)

                sr_images = model(lr_images)
                loss = criterion(sr_images, hr_images)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}")

    print("Finished Training.")

    # 모델 저장
    model_save_path = os.path.join("model", f"srcnn_model_{NUM_EPOCHS}epochs_{UPSCALE_FACTOR}upscale.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    main()
