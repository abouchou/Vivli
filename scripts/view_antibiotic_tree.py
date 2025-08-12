import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def view_antibiotic_tree():
    """Displays the antibiotic decision tree image."""
    image_path = "outputs/plots/antibiotic_decision_tree.png"
    
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} does not exist.")
        print("Please run the test_antibiotic_tree.py script first")
        return
    
    print(f"Displaying image: {image_path}")
    print("File size:", os.path.getsize(image_path), "bytes")
    
    # Load and display image
    img = mpimg.imread(image_path)
    
    plt.figure(figsize=(16, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Decision Tree for Antibiotic Recommendation', 
              fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    view_antibiotic_tree()
