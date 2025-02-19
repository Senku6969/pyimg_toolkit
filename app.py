import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageOps, ImageFont
import numpy as np
import cv2
import math
import io
import easyocr
from rembg import remove
import os
from pathlib import Path

def enhance_image(image, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0):
    img = ImageEnhance.Brightness(image).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return img

def apply_color_effect(image, effect):
    img_array = np.array(image)
    
    if effect == "Grayscale":
        return ImageOps.grayscale(image)
    elif effect == "Sepia":
        sepia_matrix = [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ]
        sepia_array = cv2.transform(img_array, np.array(sepia_matrix))
        np.clip(sepia_array, 0, 255, out=sepia_array)
        return Image.fromarray(sepia_array.astype(np.uint8))
    elif effect == "Negative":
        return ImageOps.invert(image)
    elif effect == "Black & White":
        return image.convert('L').point(lambda x: 0 if x < 128 else 255, '1')
    return image

def apply_filter(image, filter_type, strength=2):
    """Apply various image filters with customizable strength"""
    if filter_type == "Blur":
        return image.filter(ImageFilter.GaussianBlur(radius=strength))
    elif filter_type == "Contour":
        return image.filter(ImageFilter.CONTOUR)
    elif filter_type == "Emboss":
        return image.filter(ImageFilter.EMBOSS)
    elif filter_type == "Edge Enhance":
        return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    elif filter_type == "Smooth":
        return image.filter(ImageFilter.SMOOTH_MORE)
    elif filter_type == "Sharpen":
        return image.filter(ImageFilter.SHARPEN)
    return image

def add_watermark(image, text, opacity=0.5, position='bottom-right', size=None):
    """Enhanced watermark function with better positioning"""
    img = image.convert('RGBA')
    txt = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt)
    font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position
    padding = 10
    if position == 'bottom-right':
        x = img.size[0] - text_width - padding
        y = img.size[1] - text_height - padding
    elif position == 'bottom-left':
        x = padding
        y = img.size[1] - text_height - padding
    elif position == 'top-right':
        x = img.size[0] - text_width - padding
        y = padding
    elif position == 'top-left':
        x = padding
        y = padding
    else: 
        x = (img.size[0] - text_width) // 2
        y = (img.size[1] - text_height) // 2
    
    # shadow on watermark----
    # shadow_offset = 2
    # draw.text((x + shadow_offset, y + shadow_offset), text, 
    #           font=font, fill=(0, 0, 0, int(255 * opacity)))
    # draw.text((x, y), text, font=font, 
    #           fill=(255, 255, 255, int(255 * opacity)))
    
    return Image.alpha_composite(img, txt)

def detect_edges(image, threshold1=100, threshold2=200):
    """Enhanced edge detection using cv2 canny with adjustable thresholds"""
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return Image.fromarray(edges)

def remove_background_with_color(image, bg_color=(255, 255, 255)):
    output = remove(image)
    # Create new image with solid color background
    background = Image.new('RGBA', output.size, bg_color + (255,))
    # Composite the foreground onto the colored background
    final_image = Image.alpha_composite(background.convert('RGBA'), output.convert('RGBA'))
    return final_image.convert('RGB')

def enhance_pixel_quality(image, denoise_strength=10, upscale_factor=2):
    """
    Enhance image quality through denoising and upscaling using standard OpenCV methods
    denoise_strength: Strength of denoising (1-30)
    upscale_factor: Factor by which to upscale the image (1-4)
    """
    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoisingColored(
        img_array,
        None,
        denoise_strength,
        denoise_strength,
        7,
        21
    )
    # Calculate new dimensions
    height, width = denoised.shape[:2]
    new_height = int(height * upscale_factor)
    new_width = int(width * upscale_factor)
    
    # First upscale using INTER_CUBIC
    upscaled = cv2.resize(
        denoised,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC
    )
    
    # Apply sharpening
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(upscaled, -1, kernel)
    
    # Apply additional detail enhancement
    detail_enhanced = cv2.detailEnhance(sharpened, sigma_s=10, sigma_r=0.15)
    
    # Convert back to PIL Image
    return Image.fromarray(detail_enhanced)

def colorize_image(image):
    """
    Colorize a grayscale image using OpenCV's colorization model.
    """
    # Convert PIL Image to OpenCV format (numpy array)
    img_array = np.array(image)

    # Ensure the input is grayscale (if not, convert it to grayscale)
    if len(img_array.shape) == 3:  # If the image is colored (3 channels)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Ensure the image is 3-channel (convert grayscale to 3 channels)
    if len(img_array.shape) == 2:  # If grayscale (2D)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel BGR image
    
    # Load the colorization model (ensure paths are correct)
    prototxt_path = 'colorization_deploy_v2.prototxt'
    model_path = 'colorization_release_v2.caffemodel'
    kernel_path = 'pts_in_hull.npy'
    
    # Check if the model files exist
    if not all(Path(p).exists() for p in [prototxt_path, model_path, kernel_path]):
        raise FileNotFoundError("Colorization model files not found. Please download them from OpenCV's repository.")
    
    # Load the model and the points for colorization
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    pts = np.load(kernel_path)
    
    # Process model's layers and set the points
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]
    
    # Convert the image to Lab color space
    scaled = img_array.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    
    # Resize the image and split it to get the L channel
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    
    # Pass the L channel through the model to get the ab channels (color channels)
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    
    # Resize the ab channels back to the original image size
    ab = cv2.resize(ab, (img_array.shape[1], img_array.shape[0]))
    
    # Reassemble the L channel and the ab channels into a full color image
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    
    # Convert the image back to BGR color space
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    
    # Convert back to 8-bit image
    colorized = (255 * colorized).astype("uint8")
    
    return Image.fromarray(colorized)

def add_vignette(image, intensity=0.5, color=(0, 0, 0)):
    """Add vignette effect with customizable color"""
    img_array = np.array(image)
    rows, cols = img_array.shape[:2]
    
    # Generate vignette mask
    X_resultant, Y_resultant = np.meshgrid(np.arange(cols), np.arange(rows))
    centerX, centerY = cols/2, rows/2
    distance = np.sqrt((X_resultant - centerX)**2 + (Y_resultant - centerY)**2)
    
    # Normalize distances
    maxDistance = np.sqrt((centerX**2 + centerY**2))
    distance = distance/maxDistance
    
    # Create vignette mask
    vignetteMap = np.cos(distance * math.pi * intensity)
    vignetteMap = np.clip(vignetteMap, 0, 1)
    
    # Apply vignette with custom color
    for i in range(3):
        img_array[:,:,i] = img_array[:,:,i] * vignetteMap + color[i] * (1 - vignetteMap)
        
    return Image.fromarray(img_array.astype(np.uint8))

def add_frame(image, frame_width=10, frame_color=(0, 0, 0), style="Solid"):
    """Add frame with different styles"""
    if style == "Solid":
        return ImageOps.expand(image, border=frame_width, fill=frame_color)
    elif style == "Double":
        # Create double frame
        inner_width = frame_width // 2
        outer_width = frame_width
        img = ImageOps.expand(image, border=inner_width, fill=frame_color)
        return ImageOps.expand(img, border=outer_width, fill=(255, 255, 255))
    elif style == "Shadow":
        # Create shadow effect
        shadow = Image.new('RGBA', image.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rectangle([0, 0, image.size[0], image.size[1]], fill=(0, 0, 0, 100))
        shadow = shadow.rotate(5, expand=True)
        
        # Create new image with frame
        framed = ImageOps.expand(image, border=frame_width, fill=frame_color)
        final_img = Image.new('RGBA', shadow.size, (0, 0, 0, 0))
        final_img.paste(shadow, (frame_width, frame_width))
        final_img.paste(framed, (0, 0))
        return final_img.convert('RGB')
    return image

def extract_text(image, languages=['en']):
    reader = easyocr.Reader(languages)
    result = reader.readtext(np.array(image))
    return " ".join([text[1] for text in result])

def rotate_image(image, angle):
    return image.rotate(angle, expand=True)

def resize_image(image, width=None, height=None):
    if width and height:
        return image.resize((width, height))
    elif width:
        ratio = width / image.size[0]
        height = int(image.size[1] * ratio)
        return image.resize((width, height))
    elif height:
        ratio = height / image.size[1]
        width = int(image.size[0] * ratio)
        return image.resize((width, height))
    return image

# ===========================================================================================================================================================================================

# UI Configuration
st.set_page_config(
    layout="wide",
    page_title="PixelForge - AI Image Toolkit",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .stDownloadButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .stTitle {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Main Title with Logo/Icon
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.title("🎨 PixelForge")
    st.write("Your AI Powered Image Toolkit")

# Sidebar Configuration
with st.sidebar:
    st.header("📁 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "gif"])
    
    if uploaded_file:
        result = None 
        st.markdown("---")
        
        # Download Section
        st.header("💾 Export Options")
        download_format = st.selectbox(
            "Download Format",
            ["PNG", "JPEG", "WebP"],
            help="Choose the format for your downloaded image"
        )
        
        # Quality Options for JPEG
        if download_format == "JPEG":
            quality = st.slider("JPEG Quality", 1, 100, 85, help="Higher quality means larger file size")
        
        # Image Size Preview
        if "result" in locals() and result is not None:
            st.info(f"Current Image Size: {result.size[0]}x{result.size[1]} px")

    
    # Tools Section
    st.header("🛠️ Tools")
    feature = st.selectbox("Choose Tool:", [
"Basic Enhancement","IMG COLORISATION", "SUPER RESOLUTION", "BACKGROUND REMOVAL", "Color Effects", "Edge Detection", "Watermark", "Text OCR", "Rotate & Resize", "Vignette Effect", "Frame", "Filters"])
    
    # Tool-specific controls
    if feature == "Basic Enhancement":
        st.subheader("Enhancement Controls")
        brightness = st.slider("Brightness", 0.0, 2.0, 1.0, 0.1)
        contrast = st.slider("Contrast", 0.0, 2.0, 1.0, 0.1)
        saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.1)
        sharpness = st.slider("Sharpness", 0.0, 2.0, 1.0, 0.1)
        
    elif feature == "Colorize":
        st.subheader("Colorization Settings")
        st.info("This will add colors to black & white images.")
        
    elif feature == "Pixel Enhancement":
        st.subheader("Pixel Enhancement Settings")
        denoise_strength = st.slider("Denoise Strength", 1, 20, 10, help="Higher values mean stronger noise reduction")
        upscale_factor = st.slider("Upscale Factor", 1, 4, 2, help="Factor by which to increase image resolution")    
    
    elif feature == "Color Effects":
        st.subheader("Color Effect Options")
        effect = st.selectbox("Select Effect", 
            ["Grayscale", "Sepia", "Negative", "Black & White", "Colorize"])
            
    elif feature == "Edge Detection":
        st.subheader("Edge Detection Parameters")
        threshold1 = st.slider("Threshold 1", 0, 255, 100)
        threshold2 = st.slider("Threshold 2", 0, 255, 200)
        
    elif feature == "Background Removal":
        st.subheader("Background Options")
        bg_option = st.radio("Background Type", ["Transparent", "Solid Color"])
        if bg_option == "Solid Color":
            bg_color = st.color_picker("Choose Background Color", "#FFFFFF")
            
    elif feature == "Watermark":
        st.subheader("Watermark Settings")
        watermark_text = st.text_input("Watermark Text", "© 2025")
        opacity = st.slider("Opacity", 0.1, 1.0, 0.5)
        position = st.selectbox("Position", 
            ["bottom-right", "bottom-left", "top-right", "top-left", "center"])
        
        font_size = st.number_input("Font Size", 10, 300, 100)
        
    elif feature == "Text OCR":
        st.subheader("OCR Settings")
        language = st.multiselect("Select Languages", ["en", "fr", "es", "de"], default=["en"])
        st.info("Extract text from images using OCR")
        
    elif feature == "Rotate & Resize":
        st.subheader("Transform Settings")
        angle = st.slider("Rotation Angle", -180, 180, 0)
        st.write("Resize Options")
        resize_option = st.radio("Resize By", ["Width", "Height", "Both"])
        if resize_option in ["Width", "Both"]:
            width = st.number_input("Width", min_value=1, max_value=4000, value=800)
        if resize_option in ["Height", "Both"]:
            height = st.number_input("Height", min_value=1, max_value=4000, value=800)
        maintain_aspect = st.checkbox("Maintain Aspect Ratio", value=True)
        
    elif feature == "Pixel Enhancement":
        st.subheader("Enhancement Settings")
        denoise_strength = st.slider("Denoise Strength", 1, 20, 10,
            help="Higher values mean stronger noise reduction")
        upscale_factor = st.slider("Upscale Factor", 1, 4, 2,
            help="Factor by which to increase image resolution")
        
    elif feature == "Filters":
        st.subheader("Filter Options")
        filter_type = st.selectbox("Select Filter", 
            ["Blur", "Contour", "Emboss", "Edge Enhance", "Smooth", "Sharpen"])
        if filter_type == "Blur":
            blur_strength = st.slider("Blur Strength", 1, 5, 2)
            
    elif feature == "Frame":
        st.subheader("Frame Settings")
        frame_width = st.slider("Frame Width", 1, 50, 10)
        frame_color = st.color_picker("Frame Color", "#FFFFFF")
        frame_style = st.selectbox("Frame Style", ["Solid", "Double", "Shadow"])
        
    elif feature == "Vignette Effect":
        st.subheader("Vignette Settings")
        vignette_intensity = st.slider("Intensity", 0.0, 1.0, 0.5)
        vignette_color = st.color_picker("Vignette Color", "#000000")

# Main Content Area
if uploaded_file:
    # Load and process image
    image = Image.open(uploaded_file).convert("RGB")
    # Create two columns for before/after
    col1, col2 = st.columns(2)
    
    # Original Image Column
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        st.info(f"Original Size: {image.size[0]}x{image.size[1]}px")
    
    # Processed Image Column
    with col2:
        st.subheader("Processed Image")
        with st.spinner("Processing..."):
            try:
                if feature == "Text OCR":
                    text = extract_text(image)
                    result = image
                    st.image(image, use_column_width=True)
                    st.text_area("Extracted Text", text, height=200)
                else:
                    if feature == "Basic Enhancement":
                        result = enhance_image(image, brightness, contrast, saturation, sharpness)

                    elif feature == "Colorize":
                        result = colorize_image(image)
                        
                    elif feature == "Pixel Enhancement":
                        result = enhance_pixel_quality(image, denoise_strength, upscale_factor)
                    
                    elif feature == "Color Effects":
                        result = apply_color_effect(image, effect)
                        
                    elif feature == "Edge Detection":
                        result = detect_edges(image, threshold1, threshold2)
                        
                    elif feature == "Background Removal":
                        if bg_option == "Transparent":
                            result = remove_background_with_color(image)
                        else:
                            bg_color_rgb = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                            result = remove_background_with_color(image, bg_color_rgb)
                            
                    elif feature == "Watermark":
                        result = add_watermark(image, watermark_text, opacity, position, font_size)
                        
                    elif feature == "Rotate & Resize":
                        rotated = rotate_image(image, angle)
                        if resize_option == "Both":
                            result = resize_image(rotated, width, height)
                        elif resize_option == "Width":
                            result = resize_image(rotated, width=width)
                        else:
                            result = resize_image(rotated, height=height)
                            
                    elif feature == "Pixel Enhancement":
                        result = enhance_pixel_quality(image, denoise_strength, upscale_factor)
                        
                    elif feature == "Vignette Effect":
                        result = add_vignette(image, vignette_intensity)
                        
                    elif feature == "Frame":
                        result = add_frame(image, frame_width, frame_color)
                        
                    elif feature == "Filters":
                        result = apply_filter(image, filter_type)
                    
                    st.image(result, caption=f"After {feature}", use_column_width=True)
                    st.info(f"Processed Size: {result.size[0]}x{result.size[1]}px")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please try different settings or another image")
    
    # download button (saves as IO bytes - selects format and downloads)
    if 'result' in locals():
        st.sidebar.markdown("---")
        buf = io.BytesIO()
        
        # Save with appropriate format and quality
        if download_format == "JPEG":
            result.save(buf, format=download_format, quality=quality)
        else:
            result.save(buf, format=download_format)
        
        byte_im = buf.getvalue()
        
        # CSS FOR DOWNLOADING
        st.sidebar.download_button(
            label=f"📥 Download as {download_format}",
            data=byte_im,
            file_name=f"pixelforge_processed.{download_format.lower()}",
            mime=f"image/{download_format.lower()}",
            use_container_width=True
        )

else:
    # INFO SCREEN AT START (NO IMAGE UPLOADED)
    st.info("👈 Start by uploading an image from the sidebar")
    st.header("🌟 Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - ✨ Basic Image Enhancement
        - 🎨 Color Effects
        - 🖼️ Background Removal
        - ✏️ Watermark Addition
        - 📝 Text Extraction (OCR)
        - 🎨 Colorize Image
        """)
        
    with col2:
        st.markdown("""
        - 🔄 Rotate & Resize
        - 🌈 Filters
        - 🎯 Edge Detection
        - 🖌️ Vignette Effect
        - 🔲 Frame Addition
        - 📈 Pixel Enhancement
        """)