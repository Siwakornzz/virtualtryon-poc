from flask import Flask, request, make_response
from background_removal import remove_background
from parsing import parse_human
from tryon import overlay_clothing
import io

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_image():
    try:
        image_file = request.files['image']
        clothing_file = request.files['clothing']

        # Step 1: Remove background
        bg_removed = remove_background(image_file)

        # Step 2: Parse human body
        parsed_image, mask = parse_human(bg_removed)

        # Step 3: Overlay clothing
        result = overlay_clothing(parsed_image, clothing_file, mask)

        # Convert to PNG bytes
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Send as binary response
        return make_response(img_byte_arr.getvalue(), 200, {'Content-Type': 'image/png'})
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)