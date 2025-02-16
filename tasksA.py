import sqlite3
import subprocess
from dateutil.parser import parse
from datetime import datetime
import json
from pathlib import Path
import os
import requests
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

load_dotenv()

AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN')


def A1(email="21f1002773@ds.study.iitm.ac.in"):
    try:
        process = subprocess.Popen(
            ["uv", "run", "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py", email],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Error: {stderr}")
        return stdout
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error: {e.stderr}")
    

def A2(prettier_version="prettier@3.4.2", filename="/data/format.md"):
    npx_cmd = "npx"

    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' does not exist.")
        return
    
    # Check if file is writable, if not, copy to a new location
    if not os.access(filename, os.W_OK):
        print(f"Warning: No write permissions for '{filename}'. Copying to home directory...")
        new_filename = os.path.join(os.path.expanduser("~"), "format.md")
        shutil.copy(filename, new_filename)
        filename = new_filename
        print(f"File copied to '{filename}' to ensure write access.")

    # Check file encoding and convert if needed
    try:
        encoding_check = subprocess.run(["file", "-b", "--mime-encoding", filename], capture_output=True, text=True)
        encoding = encoding_check.stdout.strip()
        if encoding.lower() != "utf-8":
            print(f"Fixing encoding: {encoding} → UTF-8")
            subprocess.run(["iconv", "-f", encoding, "-t", "UTF-8", filename, "-o", filename + "_utf8"], check=True)
            shutil.move(filename + "_utf8", filename)
    except Exception as e:
        print(f"Warning: Could not check/convert file encoding: {e}")

    # Ensure Prettier is installed locally
    project_dir = os.path.expanduser("~/prettier_project")
    os.makedirs(project_dir, exist_ok=True)

    try:
        subprocess.check_call(["npm", "init", "-y"], cwd=project_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(["npm", "install", "--save-dev", "prettier"], cwd=project_dir, stdout=subprocess.DEVNULL)
        print("Prettier installed locally.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to install Prettier: {e}")
        return

    # Run Prettier and capture the full error message
    try:
        result = subprocess.run([npx_cmd, "--no-install", "prettier", "--write", filename], cwd=project_dir, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Prettier failed:\n{result.stderr}")
        else:
            print(f"Formatted {filename} successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to format {filename}: {e}")




def A3(filename='/data/dates.txt', targetfile='/data/dates-wednesdays.txt', weekday=2):
    input_file = filename
    output_file = targetfile
    weekday = weekday
    weekday_count = 0
    
    print('for A3 the parameters are', filename, targetfile, weekday)

    with open(input_file, 'r') as file:
        weekday_count = sum(1 for date in file if parse(date).weekday() == int(weekday)-1)


    with open(output_file, 'w') as file:
        file.write(str(weekday_count))


def A4(filename="/data/contacts.json", targetfile="/data/contacts-sorted.json"):
    # Load the contacts from the JSON file
    with open(filename, 'r') as file:
        contacts = json.load(file)

    # Sort the contacts by last_name and then by first_name
    sorted_contacts = sorted(contacts, key=lambda x: (x['last_name'], x['first_name']))

    # Write the sorted contacts to the new JSON file
    with open(targetfile, 'w') as file:
        json.dump(sorted_contacts, file, indent=4)

def A5(log_dir_path='/data/logs', output_file_path='/data/logs-recent.txt', num_files=10):
    log_dir = Path(log_dir_path)
    output_file = Path(output_file_path)

    # Get list of .log files sorted by modification time (most recent first)
    log_files = sorted(log_dir.glob('*.log'), key=os.path.getmtime, reverse=True)[:num_files]

    # Read first line of each file and write to the output file
    with output_file.open('w') as f_out:
        for log_file in log_files:
            with log_file.open('r') as f_in:
                first_line = f_in.readline().strip()
                f_out.write(f"{first_line}\n")

def A6(doc_dir_path='/data/docs', output_file_path='/data/docs/index.json'):
    docs_dir = doc_dir_path
    output_file = output_file_path
    index_data = {}

    # Walk through all files in the docs directory
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.md'):
                # print(file)
                file_path = os.path.join(root, file)
                # Read the file and find the first occurrence of an H1
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('# '):
                            # Extract the title text after '# '
                            title = line[2:].strip()
                            # Get the relative path without the prefix
                            relative_path = os.path.relpath(file_path, docs_dir).replace('\\', '/')
                            index_data[relative_path] = title
                            break  # Stop after the first H1
    # Write the index data to index.json
    # print(index_data)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=4)

def A7(filename='/data/email.txt', output_file='/data/email-sender.txt'):
    # Read the content of the email
    with open(filename, 'r') as file:
        email_content = file.readlines()

    sender_email = "sujay@gmail.com"
    for line in email_content:
        if "From" == line[:4]:
            sender_email = (line.strip().split(" ")[-1]).replace("<", "").replace(">", "")
            break

    # Get the extracted email address

    # Write the email address to the output file
    with open(output_file, 'w') as file:
        file.write(sender_email)

import base64
def png_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string
# def A8():
#     input_image = "data/credit_card.png"
#     output_file = "data/credit-card.txt"

#     # Step 1: Extract text using OCR
#     try:
#         image = Image.open(input_image)
#         extracted_text = pytesseract.image_to_string(image)
#         print(f"Extracted text:\n{extracted_text}")
#     except Exception as e:
#         print(f"❌ Error reading or processing {input_image}: {e}")
#         return

#     # Step 2: Pass the extracted text to the LLM to validate and extract card number
#     prompt = f"""Extract the credit card number from the following text. Respond with only the card number, without spaces:

#     {extracted_text}
#     """
#     try:
#         card_number = ask_llm(prompt).strip()
#         print(f"Card number extracted by LLM: {card_number}")
#     except Exception as e:
#         print(f"❌ Error processing with LLM: {e}")
#         return

#     # Step 3: Save the extracted card number to a text file
#     try:
#         with open(output_file, "w", encoding="utf-8") as file:
#             file.write(card_number + "\n")
#         print(f"✅ Credit card number saved to: {output_file}")
#     except Exception as e:
#         print(f"❌ Error writing {output_file}: {e}")




def preprocess_image(file_path):
    """Preprocess the image to improve OCR accuracy."""
    logger.info(f"Preprocessing image: {file_path}")

    try:
        image = Image.open(file_path).convert("L")  # Convert to grayscale
        image = image.filter(ImageFilter.SHARPEN)  # Sharpen the image
        logger.info("Enhancing Image Contrast")
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(5)  # Increase contrast
        
        # Save the preprocessed image before encoding
        processed_path = "/tmp/processed_credit_card.png"
        image.save(processed_path)
        return processed_path  # Return path of the new image

    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def png_to_base64(image_path):
    """Reads an image file and returns its Base64 encoding."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

def A8(filename='/data/credit-card.txt', image_path='/data/credit_card.png'):
    """Extract credit card number from an image and save it to a file."""
    
    # ✅ Preprocess image before OCR
    processed_image = preprocess_image(image_path)
    if not processed_image:
        logger.error("❌ Preprocessing failed. Aborting extraction.")
        return

    # ✅ Convert preprocessed image to Base64
    image_base64 = png_to_base64(processed_image)
    if not image_base64:
        logger.error("❌ Error: Unable to encode preprocessed image")
        return

    # ✅ Construct request body for AIProxy API
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the 16-digit credit card number from this image. "
                                "Return ONLY the number, without spaces, hyphens, or any other characters."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }

    try:
        # ✅ Make the API request
        response = requests.post("http://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
                                 headers=headers, data=json.dumps(body))
        response.raise_for_status()
        result = response.json()

        # ✅ Debugging: Log the full response
        logger.info("📝 AI Response: %s", json.dumps(result, indent=2))

        # ✅ Extract the credit card number
        if 'choices' in result and result['choices']:
            raw_output = result['choices'][0]['message']['content']
            card_number = "".join(filter(str.isdigit, raw_output))  # Keep only digits

            # ✅ Ensure exactly 16 digits are extracted
            if len(card_number) != 16:
                logger.warning(f"⚠️ Warning: Extracted number '{card_number}' might be incorrect.")
                return

            # ✅ Write the extracted card number to the file
            with open(filename, 'w') as file:
                file.write(card_number)
            logger.info(f"✅ Success: Extracted credit card number '{card_number}' saved to {filename}")

        else:
            logger.error(f"❌ Error: Unexpected API response format! Check the response above.")

    except Exception as e:
        logger.error(f"❌ API Request Failed: {e}")

def A9(filename='/data/comments.txt', output_filename='/data/comments-similar.txt'):
    # Read comments
    with open(filename, 'r') as f:
        comments = [line.strip() for line in f.readlines()]

    # Get embeddings for all comments
    embeddings = [get_embedding(comment) for comment in comments]

    # Find the most similar pair
    min_distance = float('inf')
    most_similar = (None, None)

    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            distance = cosine(embeddings[i], embeddings[j])
            if distance < min_distance:
                min_distance = distance
                most_similar = (comments[i], comments[j])

    # Write the most similar pair to file
    with open(output_filename, 'w') as f:
        f.write(most_similar[0] + '\n')
        f.write(most_similar[1] + '\n')

def A10(filename='/data/ticket-sales.db', output_filename='/data/ticket-sales-gold.txt', query="SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'"):
    # Connect to the SQLite database
    conn = sqlite3.connect(filename)
    cursor = conn.cursor()

    # Calculate the total sales for the "Gold" ticket type
    cursor.execute(query)
    total_sales = cursor.fetchone()[0]

    # If there are no sales, set total_sales to 0
    total_sales = total_sales if total_sales else 0

    # Write the total sales to the file
    with open(output_filename, 'w') as file:
        file.write(str(total_sales))

    # Close the database connection
    conn.close()
