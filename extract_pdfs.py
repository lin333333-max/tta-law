import pypdf
import sys

# Extract PCL paper
print("===== PCL Paper =====", file=sys.stderr)
with open('paper/Test-Time Adaptation with Perturbation Consistency Learning.pdf', 'rb') as f:
    pdf = pypdf.PdfReader(f)
    text_parts = []
    for i in range(min(10, len(pdf.pages))):
        text = pdf.pages[i].extract_text()
        text_parts.append(f"\n--- Page {i+1} ---\n{text}")

    with open('paper/pcl_extract.txt', 'w', encoding='utf-8') as out:
        out.write('\n'.join(text_parts))
    print(f"PCL: Extracted {len(pdf.pages)} pages to pcl_extract.txt", file=sys.stderr)

# Extract K-LJP paper
print("\n===== K-LJP Paper =====", file=sys.stderr)
with open('paper/K-LJP.pdf', 'rb') as f:
    pdf = pypdf.PdfReader(f)
    text_parts = []
    for i in range(len(pdf.pages)):
        text = pdf.pages[i].extract_text()
        text_parts.append(f"\n--- Page {i+1} ---\n{text}")

    with open('paper/kljp_extract.txt', 'w', encoding='utf-8') as out:
        out.write('\n'.join(text_parts))
    print(f"K-LJP: Extracted {len(pdf.pages)} pages to kljp_extract.txt", file=sys.stderr)

print("\nDone! Check pcl_extract.txt and kljp_extract.txt", file=sys.stderr)
