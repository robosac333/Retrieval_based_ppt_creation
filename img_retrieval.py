import os
from pdf2image import convert_from_path
from PIL import Image
import fitz  # PyMuPDF

# ---------------------------------------------------------------
# 1. Convert PDFs to page images (once)
# ---------------------------------------------------------------
def convert_pdfs_to_images(pdf_folder, output_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    all_images = {}
    file_names = {}

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for doc_id, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        base_name = os.path.splitext(pdf_file)[0]

        # Skip conversion if images already exist
        existing = [f for f in os.listdir(output_folder)
                    if f.startswith(base_name) and f.endswith(".png")]
        if existing:
            print(f"Skipping {pdf_file}, already converted.")
            all_images[doc_id] = [os.path.join(output_folder, f) for f in sorted(existing)]
            file_names[doc_id] = pdf_file
            continue

        images = convert_from_path(pdf_path)
        print(f"Converted {pdf_file} to {len(images)} images.")

        image_paths = []
        for i, img in enumerate(images):
            image_filename = f"{base_name}_page_{i + 1}.png"
            image_path = os.path.join(output_folder, image_filename)
            img.save(image_path, "PNG")
            image_paths.append(image_path)

        all_images[doc_id] = image_paths
        file_names[doc_id] = pdf_file

    return all_images, file_names

# ---------------------------------------------------------------
# 2. Extract embedded images from PDF
# ---------------------------------------------------------------
def extract_images_from_pdf(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img_count = 0
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            img_filename = f"{base_name}_img_{img_count + 1}.png"
            img_path = os.path.join(output_folder, img_filename)

            with open(img_path, "wb") as img_file:
                img_file.write(image_bytes)

            img_count += 1

    print(f"Extracted {img_count} images from {pdf_path}")

# ---------------------------------------------------------------
# 3. Map RAG search results to image paths
# ---------------------------------------------------------------
def get_grouped_images(results, all_images):
    grouped_images = []
    for result in results:
        doc_id = result["doc_id"]
        page_num = result["page_num"]  # 1-indexed
        grouped_images.append(all_images[doc_id][page_num - 1])
    return grouped_images


if __name__ == "__main__":
    
    pdf_folder = "/nfshomes/sjd3333/Retrieval_based_ppt_creation_copy/input_directory"
    output_pages = "/nfshomes/sjd3333/Retrieval_based_ppt_creation_copy/image_output"
    output_embeds = "/nfshomes/sjd3333/Retrieval_based_ppt_creation_copy/embedded_images"

    all_images, file_names = convert_pdfs_to_images(pdf_folder, output_pages)

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            extract_images_from_pdf(pdf_path, output_embeds)

    dummy_results = [{"doc_id": 0, "page_num": 1}]
    selected = get_grouped_images(dummy_results, all_images)
    print(f"\nGrouped image path(s): {selected}")
