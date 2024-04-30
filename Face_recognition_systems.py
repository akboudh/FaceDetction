import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import face_recognition
import time

# Function to load dataset
def load_dataset():
    dataset = {}
    if os.path.exists("dataset.npy"):
        dataset = np.load("dataset.npy", allow_pickle=True).item()
    return dataset

# Function to save dataset
def save_dataset(dataset):
    np.save("dataset.npy", dataset)

# Function to insert a new face to the dataset
def insert_face(dataset):
    name = input("Enter the name of the person: ")
    image_path = input("Enter the path of the image: ")

    if not os.path.exists(image_path):
        print("Image file not found!")
        return

    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if len(face_encodings) == 0:
        print("No face found in the image!")
        return

    if name in dataset:
        dataset[name].append((image_path, face_encodings[0]))
    else:
        dataset[name] = [(image_path, face_encodings[0])]

    save_dataset(dataset)
    print("Face inserted successfully!")

# Function to calculate similarity score
def calculate_similarity(embedding1, embedding2):
    # Calculate cosine similarity between embeddings
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    # Convert cosine similarity to percentage
    similarity_percentage = similarity * 100
    return similarity_percentage

# Function to check if a person is present in the dataset
# Function to check if a person is present in the dataset
def check_presence(dataset):
    image_path = input("Enter the path of the image to check: ")

    if not os.path.exists(image_path):
        print("Image file not found!")
        return

    unknown_image = face_recognition.load_image_file(image_path)
    unknown_encoding = face_recognition.face_encodings(unknown_image)

    if len(unknown_encoding) == 0:
        print("No face found in the image!")
        return

    max_similarity_score = 0
    best_match = None

    for name, images_encodings in dataset.items():
        for img_path, known_encoding in images_encodings:
            # Compare the face in the image with the faces in the dataset
            match = face_recognition.compare_faces([known_encoding], unknown_encoding[0])
            if match[0]:
                similarity_score = calculate_similarity(known_encoding, unknown_encoding[0])
                if similarity_score > max_similarity_score:
                    max_similarity_score = similarity_score
                    best_match = (name, img_path)

    if best_match:
        name, img_path = best_match
        print(f"Person {name} found in the dataset with maximum similarity score: {max_similarity_score:.2f}%")
        show_images(img_path, image_path, f"Person {name}'s Image", "Given Image", max_similarity_score)
    else:
        print("Person not found in the dataset.")


# Function to delete a face from the dataset
def delete_face(dataset):
    name = input("Enter the name of the person to delete: ")

    if name in dataset:
        del dataset[name]
        save_dataset(dataset)
        print("Person deleted from the dataset.")
    else:
        print("Person not found in the dataset.")

# Function to display images
def show_images(image_path1, image_path2, title1, title2, similarity_score=None):
    image1 = cv2.imread(image_path1)
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    image2 = cv2.imread(image_path2)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image1_rgb)
    axes[0].set_title(title1)
    axes[0].axis('off')

    axes[1].imshow(image2_rgb)
    axes[1].set_title(title2)
    axes[1].axis('off')

    if similarity_score is not None:
        plt.suptitle(f"Similarity Score: {similarity_score:.2f}%", fontsize=14)

    plt.show(block=False)  # Show the plot without blocking
    # Wait for 3 seconds
    time.sleep(3)
    plt.close()  # Close the plot

# Function to automatically add people from a dataset directory
def add_people_from_dataset(dataset):
    dataset_root = "/content/MYDATASET1"  # Adjust this path to match your dataset root directory
    for person_name in os.listdir(dataset_root):
        person_folder = os.path.join(dataset_root, person_name)
        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                # Skip .DS_Store files
                if filename == ".DS_Store":
                    continue
                image_path = os.path.join(person_folder, filename)
                if os.path.isfile(image_path):
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    if len(face_encodings) > 0:
                        encoding = face_encodings[0]
                        if person_name in dataset:
                            dataset[person_name].append((image_path, encoding))
                        else:
                            dataset[person_name] = [(image_path, encoding)]
    save_dataset(dataset)
    print("People added from dataset successfully!")

# Function to list the number of people in the dataset along with the number of encodings for each person
def list_people_in_dataset(dataset):
    print("Number of people in the dataset:", len(dataset))
    print("List of people:")
    for idx, (person_name, encodings) in enumerate(dataset.items(), 1):
        num_encodings = len(encodings)
        print(f"{idx}. {person_name}: {num_encodings} encodings")

# Main function
def main():
    dataset = load_dataset()

    while True:
        print("\nMenu:")
        print("1) Insert a new face to the dataset")
        print("2) Check if person is present in the dataset")
        print("3) Delete from dataset")
        print("4) Add people from dataset automatically")
        print("5) List number of people in dataset")
        print("6) Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            insert_face(dataset)
        elif choice == '2':
            check_presence(dataset)
        elif choice == '3':
            delete_face(dataset)
        elif choice == '4':
            add_people_from_dataset(dataset)
        elif choice == '5':
            list_people_in_dataset(dataset)
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()

