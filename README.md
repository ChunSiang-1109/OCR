# What is OCR?

OCR (Optical Character Recognition) is a technology used to recognize and extract text from images or scanned documents. It converts visual text (like what's found in photos or scans) into machine-readable characters. OCR is commonly used in document digitization, automated data entry, and reading information from forms, invoices, product labels, etc.

This Python script uses PaddleOCR to extract text from label images (like equipment or product labels), find specific pieces of information (like serial number, model, manufacturer), and save:

1.The image with bounding boxes around recognized text, and

2.A JSON file with the extracted information.
