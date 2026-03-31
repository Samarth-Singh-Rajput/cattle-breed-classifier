# Cattle/Buffalo Breed Classifier

This project uses a ConvNeXt Tiny model to classify cattle and buffalo breeds from images.

## Project Structure
```
cattle-breed-classifier/
│
├── models/
│   └── convnext_tiny_best_breed-classifier.pth
│
├── notebooks/
│   └── convon_tiny_breed-classifier.ipynb
│
├── results/
│   ├── confusion_epoch_1.png
│   ├── confusion_epoch_2.png
│   ├── confusion_epoch_3.png
│   ├── confusion_epoch_4.png
│   └── confusion_epoch_5.png
│
├── src/
│   └── inference.py
│
├── requirements.txt
└── README.md
```
## How to Use

1. **Install dependencies:**
   ```zsh
   pip install -r requirements.txt
2. **Train the model:**

See notebooks/convon_tiny_breed-classifier.ipynb for training and evaluation.

3. **Run inference:**

   python src/inference.py path/to/your/image.jpg

4.**Results:**
  
  Confusion matrices for the first 5 epochs are in the results/ folder.

License
MIT License

