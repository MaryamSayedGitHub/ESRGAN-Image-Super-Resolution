import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from RRDBNet_arch import RRDBNet

# Load model
@st.cache_resource
def load_model():
    model = RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load("/content/RRDB_PSNR_x4.pth"), strict=True)
    model.eval()
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

model = load_model()

def enhance_image(image: Image.Image):
    img = np.array(image).astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    with torch.no_grad():
        output = model(img).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0)) * 255
    return Image.fromarray(np.uint8(output))

# Streamlit UI
# ✅ Set page config must come first!
st.set_page_config(page_title="ESRGAN Super Resolution", layout="centered")

# Now other Streamlit stuff
st.title("ESRGAN Image Super Resolution")
st.markdown("Upload a low-resolution image to upscale it using RRDBNet (ESRGAN).")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Enhance Image"):
        with st.spinner("Enhancing image..."):
            enhanced = enhance_image(image)
        st.image(enhanced, caption="Enhanced Image", use_column_width=True)

        # Download
        from io import BytesIO
        buf = BytesIO()
        enhanced.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label="📥 Download Enhanced Image",
            data=byte_im,
            file_name="enhanced.png",
            mime="image/png"
        )
