def ExtractTextFromImage(image_path):
    # Load the image using PIL
    image = screenshot

    # Convert the image to grayscale
    grayscale_image = image.convert("L")

    # Use pytesseract to extract text from the grayscale image
    extracted_text = pytesseract.image_to_string(grayscale_image)

    # Return the extracted text
    return extracted_text