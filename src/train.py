from loader import *
from model import *
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f"Using device: {device}")

MODEL_SAVE_PATH = r"..\results\models\model.pt"

train_loader, _, train_labels_array = get_dataloaders()

model = AlzhSpeechNN().to(device)
weights = compute_class_weight('balanced', classes =np.array([0,1]), y = train_labels_array)
criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor(weights, dtype = torch.float32).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay = 1e-5)
scheduler = StepLR(optimizer, step_size = 30, gamma = 0.5)
num_epochs = 60

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted==labels).sum().item()
        total += labels.size(0)
    scheduler.step()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch + 1}/{num_epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}% | LR: {current_lr:.6f}")

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Training complete. Model saved.")