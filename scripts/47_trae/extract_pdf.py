
import PyPDF2
import os

def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            print(f"Total Pages: {num_pages}")
            
            # Extract text from pages 5 to 10
            for i in range(5, min(10, num_pages)):
                page = reader.pages[i]
                text = page.extract_text()
                print(f"--- Page {i+1} ---")
                print(text)
                print("\n")
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    extract_text_from_pdf("47_AMP_GNN.pdf")
