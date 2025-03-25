# Load the dataset

# ---------------------------------------------------------------
# 1. Convert PDFs to images
# ---------------------------------------------------------------
import os
from pdf2image import convert_from_path
import time
import numpy as np
import cv2
from dotenv import load_dotenv
from PIL import Image
from byaldi import RAGMultiModalModel
from transformers import pipeline, AutoProcessor

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

def convert_pdfs_to_images(pdf_folder, output_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_images = {}
    file_names = {}

    for doc_id, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        images = convert_from_path(pdf_path)
        print(f"Converted {pdf_file} to {len(images)} images.")

        image_filenames = []
        for i, img in enumerate(images):
            image_filename = f"{pdf_file.replace('.pdf', '')}_page_{i + 1}.png"
            image_path = os.path.join(output_folder, image_filename)
            img.save(image_path, "PNG")  # Save images properly
            image_filenames.append(image_path)

        all_images[doc_id] = image_filenames  # Store list of images per document
        file_names[doc_id] = pdf_file

    return all_images, file_names


pdf_folder = "/nfshomes/sjd3333/Retrieval_based_ppt_creation/pdfs"
image_output_folder = "/nfshomes/sjd3333/Retrieval_based_ppt_creation/img_output"

all_images, file_names = convert_pdfs_to_images(pdf_folder, image_output_folder)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# 2. Initialize the ColPali Multimodal Document Retrieval Model
# ---------------------------------------------------------------
docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")

# Create index for the PDF folder
docs_retrieval_model.index(input_path=pdf_folder, index_name="image_index", store_collection_with_index=False, overwrite=True)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# 3. Retrieve Documents with the Document Retrieval Model - ColPali
# ---------------------------------------------------------------

def get_relevant_docs(text_query):
    results = docs_retrieval_model.search(text_query, k=3)  # Retrieve top 3 documents
    return results

# ---------------------------------------------------------------
# 4. Initialize the Visual Language Model 
# ---------------------------------------------------------------
model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline("image-to-text", model=model_id, device=0)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# 5. Chat with Chatbot
# ---------------------------------------------------------------
while True:
    text_query = input("You: ")
    if text_query.lower() == "exit":
        print("Bye Bye..!")
        break

    results = get_relevant_docs(text_query)
    print(f"No of Relevant Documents Retrieved: {len(results)}")

    # 🔹 Debugging: Print results to check if retrieval works
    print(f"Results: {results}")

    # ✅ Prevent crash if no results are found
    if not results:
        print("⚠ No relevant documents found. Try another query.")
        continue

    # ✅ Check if PNG images exist in the folder
    images_path = image_output_folder
    png_files = [f for f in os.listdir(images_path) if f.endswith('.png')]

    if not png_files:
        print("⚠ No PNG images found in the folder!")
        continue

    # ✅ Ensure doc_id is within valid range
    doc_id = results[0].doc_id  # Access attributes directly
    page_num = results[0].page_num


    if doc_id is None or page_num is None:
        print(f"⚠ Invalid document ID or page number. Skipping retrieval.")
        continue

    # ✅ Get the correct image file for the retrieved document & page
    image_files_for_doc = all_images.get(doc_id, [])
    
    if page_num - 1 >= len(image_files_for_doc):
        print(f"⚠ Page {page_num} does not exist for document {doc_id}. Skipping.")
        continue

    image_path = image_files_for_doc[page_num - 1]  # Page numbers are 1-indexed
    try:
        image = Image.open(image_path)
        print(f"Extracted Image: {image_path}")
    except Exception as e:
        print(f"⚠ Error loading image: {e}")
        continue

    # Chat template for Qwen2 Model
    chat_template = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image 
                },
                {"type": "text", "text": text_query},
            ],
        }
    ]

    # Query the model
    start_time = time.time()  # Track response time

    processor = AutoProcessor.from_pretrained(model_id)
    prompt = processor.apply_chat_template(chat_template, add_generation_prompt=True)

    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

    response_time = time.time() - start_time  # Compute response time

    print("----------------------------------------------------------------------")
    print("Llava Bot:", outputs[0]["generated_text"].split("ASSISTANT:")[1])  # Display response
    print("----------------------------------------------------------------------")
    print(f"Time Taken to Respond: {response_time:.2f} seconds")  # Show res