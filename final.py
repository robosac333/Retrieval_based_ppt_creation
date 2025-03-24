from txt_retrieval import create_vector_db, TextFileLoader
from img_retrieval import convert_pdfs_to_images, extract_images_from_pdf, get_grouped_images
import os
import time
# from PIL import Image
import numpy as np
import cv2    
from Chatbot.Mistral_7b import retrieve_faiss, retrieve_context, generate_answer
from transformers import pipeline, AutoProcessor
from byaldi import RAGMultiModalModel

# Define the path to the dataset directory containing PDF and text files
data_path = "/nfshomes/sjd3333/Retrieval_based_ppt_creation_copy/input_directory" 

# Define the path where the FAISS vector database will be saved
Db_faiss_path = "/nfshomes/sjd3333/Retrieval_based_ppt_creation_copy/vector_db"

# To save images in case
images_folder = '/nfshomes/sjd3333/Retrieval_based_ppt_creation_copy/image_output'

all_images, file_names = convert_pdfs_to_images(data_path, images_folder)
# # ---------------------------------------------------------------


# ---------------------------------------------------------------
# 2. Initialize the ColPali Multimodal Document Retrieval Model
# ---------------------------------------------------------------
docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
# input the path to the data folder containing the PDFs 
# This creates the indexes of pdfs and assingn a unique name using index_name
docs_retrieval_model.index(input_path=data_path, index_name="image_index", store_collection_with_index=False, overwrite=True)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# 3. Retrieving Documents with the Document Retrieval Model - ColPali
# ---------------------------------------------------------------
def get_relevant_docs(text_query):
    results = docs_retrieval_model.search(text_query, k=2)      # k - Number of relevant document pages to retrieve
    return results

# ---------------------------------------------------------------
# 4. Initialize the Visual Language Model 
# ---------------------------------------------------------------
model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline("image-to-text", model=model_id, device=0)
# ---------------------------------------------------------------

create_vector_db(data_path, Db_faiss_path)  # Call the function to create and save the vector database

# extract_images_from_pdf(data_path, images_folder)

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
    png_files = [f for f in os.listdir(images_folder) if f.endswith('.png')]

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

    image_pil = image_files_for_doc[page_num - 1]  # This is already a PIL Image
    try:
        # Convert PIL Image to numpy array (no need for OpenCV here)
        image = image_pil
        # If the image is in RGB mode, no need to convert
        print(f"Extracted Image from PIL object")
    except Exception as e:
        print(f"⚠ Error processing image: {e}")
        continue

    # Chat template for LLava Model
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

    # ---------------------------------------------------------------
    # 5. Chat with Chatbot
    # ---------------------------------------------------------------

    db = retrieve_faiss(Db_faiss_path)

    context = retrieve_context(text_query, db)  # Retrieve relevant context documents
    image_context = outputs[0]["generated_text"].split("ASSISTANT:")[1]
    print(f"Image Context: {image_context}")

    answer = generate_answer(text_query, context, image_context=image_context)  # Generate the model's answer

    response_time = time.time() - start_time  # Calculate how long the response took.

    print("Mistral Bot:", answer)  # Display the response.
    print(f"Time Taken to Respond: {response_time:.2f} seconds")  # Display the response time.
    print("----------------------------------------------------------------------")
    # ---------------------------------------------------------------
