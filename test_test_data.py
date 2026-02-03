import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(42)

def preprocess_image(img_path, target_size=(64, 64)):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Slika nije pronađena: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return img

def main():
    model_path = 'traffic_sign_model.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model nije pronađen na putanji: {model_path}")
    print(f"Učitavam model iz: {model_path}")
    model = load_model(model_path)

    csv_path = 'Test.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Datoteka Test.csv nije pronađena u trenutnom direktoriju.")
    test_df = pd.read_csv(csv_path)
    if 'ClassId' not in test_df.columns or 'Path' not in test_df.columns:
        raise KeyError("U Test.csv očekujem stupce 'ClassId' i 'Path'.")

    test_dir = 'Test'  
    test_images = []
    test_labels = []

    for idx, row in test_df.iterrows():
        rel_path = row['Path']               
        true_label = int(row['ClassId'])    
        img_filename = os.path.basename(rel_path)
        img_path = os.path.join(test_dir, img_filename)

        if os.path.exists(img_path):
            test_images.append(img_path)
            test_labels.append(true_label)
        else: 
            print(f"[Upozorenje] Slika nije pronađena, preskačem: {img_path}")

    if len(test_images) == 0:
        print("Nijedna slika nije pronađena u mapi Test prema informacijama iz Test.csv.")
        return   
    correct_predictions = 0
    accuracies = []
    predictions_count = []

    random_indices = np.random.permutation(len(test_images))
    test_images_shuffled = [test_images[i] for i in random_indices]
    test_labels_shuffled = [test_labels[i] for i in random_indices]

    print(f"Ukupno slika za testiranje: {len(test_images)}")
    for i, (img_path, true_label) in enumerate(tqdm(zip(test_images_shuffled, test_labels_shuffled), 
                                                   total=len(test_images)), start=1):
        try:
            img = preprocess_image(img_path, target_size=(64, 64))
        except Exception as e:
            print(f"[Greška] Ne mogu učitati sliku {img_path}: {e}")
            continue

        img_batch = np.expand_dims(img, axis=0)
        prediction = model.predict(img_batch, verbose=0)[0]
        predicted_class = np.argmax(prediction)

        if predicted_class == true_label:
            correct_predictions += 1

        current_accuracy = (correct_predictions / i) * 100
        accuracies.append(current_accuracy)
        predictions_count.append(i)

    plt.figure(figsize=(10, 6))
    plt.plot(predictions_count, accuracies, color='blue', linewidth=2)
    plt.xlabel('Broj predikcija', fontsize=12)
    plt.ylabel('Točnost (%)', fontsize=12)
    plt.title('Točnost modela kroz redom testirane slike', fontsize=14)
    plt.grid(True)

    final_accuracy = accuracies[-1]
    plt.text(0.02, 0.95, f'Konačna točnost: {final_accuracy:.2f}%', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8), 
             fontsize=12)

    output_graph = 'test_accuracy_graph.png'
    plt.savefig(output_graph, dpi=150, bbox_inches='tight')
    plt.close()

    print("\n----------------------------------------")
    print(f"Konačna točnost na svim test slikama: {final_accuracy:.2f}%")
    print(f"Broj slika obrađenih za test: {len(predictions_count)}")
    print(f"Graf je spremljen kao: '{output_graph}'")
    print("----------------------------------------")

if __name__ == '__main__':
    main()
