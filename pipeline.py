import torch
import torch.nn as nn
import kornia.augmentation as K

class MockVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, 256)
        )
        self.text_projection = nn.Linear(256, 128)

    def forward(self, image_tensor):
        features = self.vision_encoder(image_tensor)
        embeddings = self.text_projection(features)
        return embeddings

class KorniaVLMPipeline(nn.Module):
    def __init__(self, vlm_model):
        super().__init__()
        self.vlm = vlm_model
        self.augmentations = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(0.1, 0.1, 0.1, 0.1),
            K.Resize((224, 224)),
            K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), 
                        std=torch.tensor([0.229, 0.224, 0.225]))
        )

    def forward(self, input_image):
        augmented_image = self.augmentations(input_image)
        output = self.vlm(augmented_image)
        return output

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dummy_input = torch.rand(1, 3, 512, 512).to(device)
    
    base_model = MockVLM().to(device)
    pipeline = KorniaVLMPipeline(base_model).to(device)
    
    output_embeddings = pipeline(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output embeddings shape: {output_embeddings.shape}")