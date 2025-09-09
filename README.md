# 🍔 Calorie & Nutrition Predictor

An AI-powered application that predicts **calories and nutritional values** (carbohydrates, proteins, fats, etc.) of food items directly from an **image input**.  
Built with **Deep Learning (CNNs)** for food recognition and a **nutrition mapping pipeline** for macro/micro nutrient estimation.

---

## 🚀 Features
- 📸 Upload an image of a food item to get calorie & nutrient predictions  
- 🍕 Supports multiple common food categories (e.g., pizza, burger, fries, rice, etc.)  
- 🔍 Pretrained **CNN-based food classification model** (transfer learning on Food101/custom dataset)  
- 📊 Nutrition estimation using a food-to-nutrients mapping dataset (USDA/other sources)  
- 🖥️ Easy-to-use API/Notebook interface  

---

## 📂 Project Structure
```
calorie_predictor/
│── data/                 # Datasets (Food Images + Nutrition DB)
│── notebooks/            # Training & evaluation notebooks
│── models/               # Saved PyTorch/TensorFlow models
│── src/                  # Source code
│   ├── preprocessing.py  # Image preprocessing
│   ├── model.py          # CNN/Transfer Learning architecture
│   ├── predict.py        # Prediction pipeline
│   ├── nutrition_map.py  # Food → Nutrient mapping
│── app/                  # Optional: API or frontend code
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
```

---

## ⚙️ Tech Stack
- **Python**, **PyTorch/TensorFlow/Keras** for Deep Learning  
- **CNN / Transfer Learning (ResNet, EfficientNet, VGG16, etc.)** for food recognition  
- **Pandas, NumPy** for nutrition data handling  
- **Matplotlib/Seaborn** for visualization  
- **FastAPI/Flask (optional)** for API deployment  

---

## 📊 Workflow
1. **Image Input** → Upload a food item image  
2. **Food Recognition** → CNN model classifies the food category  
3. **Nutrient Mapping** → Match predicted class with nutrition database  
4. **Results Output** → Calories, proteins, carbs, fats, etc.  

Example output:  
```json
{
  "food_item": "Pizza",
  "calories": 285,
  "protein_g": 12,
  "carbs_g": 36,
  "fat_g": 10
}
```

---

## 🧑‍💻 Installation & Usage
```bash
# Clone the repo
git clone https://github.com/your-username/calorie_predictor.git
cd calorie_predictor

# Create environment & install dependencies
pip install -r requirements.txt
```

Run predictions:
```bash
python src/predict.py --image samples/pizza.jpg
```

---

## 📈 Results
- Achieved **~85% Top-1 Accuracy** on validation food dataset  
- Nutrition predictions aligned with standard nutrition databases  

---

## 📌 Future Improvements
- Expand dataset for more food items  
- Improve multi-food dish detection (multiple items in one image)  
- Add mobile/Android app integration  
- Support real-time camera-based predictions  

---

## 🤝 Contributing
Contributions are welcome!  
- Fork the repo  
- Create a feature branch  
- Submit a Pull Request 🚀  

---

## 📜 License
This project is licensed under the **MIT License**.  

---

## 🙌 Acknowledgements
- [Food101 Dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)  
- [USDA Food Composition Databases](https://fdc.nal.usda.gov/)  
- Transfer Learning models from **PyTorch** and **TensorFlow Hub**  
