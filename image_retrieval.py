
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
import fitz

def convert_pdfs_to_images(pdf_folder, output_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    all_images = {}
    file_names = {}

    for doc_id, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        images = convert_from_path(pdf_path)
        print(f"Converted {pdf_file} to {len(images)} images.")
        for i, img in enumerate(images):
            image_filename = f"{pdf_file.replace('.pdf', '')}_page_{i + 1}.png"
            image_path = os.path.join(output_folder, image_filename)
            img.save(image_path, "PNG")  # Save images properly
            # image_filenames.append(image_path)

        all_images[doc_id] = images
        file_names[doc_id] = pdf_file

    return all_images, file_names
# ---------------------------------------------------------------



# ----------------------------------------------------------------
# Seperate the images from the pdf
# ----------------------------------------------------------------
def extract_images_from_pdf(pdf_path, output_folder):
    # Open the PDF
    doc = fitz.open(pdf_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img_count = 0  # To count the number of images extracted

    # Iterate through the pages of the PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Get the page
        
        # Extract images from the page
        image_list = page.get_images(full=True)
        
        # Loop through each image on the page
        for img_index, img in enumerate(image_list):
            xref = img[0]  # xref of the image
            base_image = doc.extract_image(xref)  # Extract the image as a dictionary
            image_bytes = base_image["image"]  # Get the image bytes
            
            # Create an output path for saving the image
            img_filename = f"image_{img_count + 1}.png"
            img_path = os.path.join(output_folder, img_filename)
            
            # Save the image to the output folder
            with open(img_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            img_count += 1

    print(f"Extracted {img_count} images and saved them to {output_folder}")



# Get the corresponding images of the retrieved documents
def get_grouped_images(results, all_images):
    grouped_images = []

    for result in results:
        doc_id = result["doc_id"]
        page_num = result["page_num"]
        grouped_images.append(
            all_images[doc_id][page_num - 1]
        )  # page_num are 1-indexed, while doc_ids are 0-indexed. Source https://github.com/AnswerDotAI/byaldi?tab=readme-ov-file#searching

    return grouped_images

