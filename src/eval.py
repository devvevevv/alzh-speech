import numpy as np
from loader import  *
from model import *
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SAVED_MODEL_PATH = "../results/models/model.pt"

_, test_loader, _ = get_dataloaders()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlzhSpeechNN().to(device)
model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, outputs = inputs.to(device), model(inputs)
        _, predicted = torch.max(outputs, 1)
        for p in predicted:
            all_preds.append(p.item())
        for l in labels:
            all_labels.append(l.item())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {accuracy*100:.2f}%\n")

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Control", "Dementia"]))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))