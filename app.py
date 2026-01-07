# import torch.nn.functional as F
# import streamlit as st
# import torch
# import torchvision.transforms as transforms
# import torchvision.models as models
# import torch.nn as nn
# from PIL import Image

# st.title("🦐 Prawn Disease Detection App")

# @st.cache_resource
# def load_model():
#     model = models.efficientnet_b0(weights=None)
#     model.classifier[1] = nn.Linear(
#         model.classifier[1].in_features, 2
#     )

#     state_dict = torch.load(
#         "model/best_model.pt",
#         map_location="cpu"
#     )
#     model.load_state_dict(state_dict)
#     model.eval()
#     return model

# model = load_model()
# st.success("✅ Classification model loaded successfully")
# st.header("📤 Upload Prawn Image")

# uploaded_file = st.file_uploader(
#     "Choose a prawn image",
#     type=["jpg", "jpeg", "png"]
# )

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)



# # Image preprocessing (same as training)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# if image is not None:
#     st.subheader("🔍 Prediction")

#     img_tensor = transform(image).unsqueeze(0)

#     with torch.no_grad():
#         outputs = model(img_tensor)
#         probs = torch.sigmoid(outputs).squeeze().tolist()

#     bg_prob = probs[0]
#     wssv_prob = probs[1]
#     if bg_prob > 0.6 and wssv_prob > 0.6:

#         prediction = "🟠 Mixed Infection (BG + WSSV)"
#     elif bg_prob >= wssv_prob:

#         prediction = "🟢 Black Gill Disease (BG)"
#     else:

#         prediction = "🔴 White Spot Syndrome Virus (WSSV)"
#     st.write(f"**BG Probability:** {bg_prob:.2f}")
#     st.write(f"**WSSV Probability:** {wssv_prob:.2f}")
#     st.success(f"**Prediction:** {prediction}")














# import streamlit as st
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import torchvision.models as models
# from PIL import Image

# st.set_page_config(
#     page_title="Prawn Disease Detection",
#     page_icon="🦐",
#     layout="centered"
# )

# st.title("🦐 Prawn Disease Detection App")
# st.write("Upload or scan a prawn image to detect diseases")

# MODEL_PATH = "model/best_model.pt"

# @st.cache_resource
# def load_model():
#     model = models.efficientnet_b0(weights=None)
#     model.classifier[1] = nn.Linear(
#         model.classifier[1].in_features, 2
#     )

#     state_dict = torch.load(MODEL_PATH, map_location="cpu")
#     model.load_state_dict(state_dict, strict=True)
#     model.eval()
#     return model

# model = load_model()
# st.success("✅ Classification model loaded successfully")


# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])


# st.subheader("📤 Upload or Scan Image")

# uploaded_file = st.file_uploader(
#     "Upload prawn image",
#     type=["jpg", "jpeg", "png"]
# )

# camera_image = st.camera_input("📷 Scan using camera")

# image = None

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
# elif camera_image:
#     image = Image.open(camera_image).convert("RGB")


# if image:
#     st.image(image, caption="Input Image", use_container_width=True)

#     img_tensor = transform(image).unsqueeze(0)

#     with torch.no_grad():
#         outputs = model(img_tensor)
#         probs = torch.sigmoid(outputs)[0]

#     bg_prob = probs[0].item()
#     wssv_prob = probs[1].item()


#     if bg_prob > 0.6 and wssv_prob > 0.6:
#         disease = "WSSV_BG"
#         st.error("🚨 Mixed Infection Detected (BG + WSSV)")
#         recommendation = "Immediate expert intervention required."
#     elif wssv_prob > bg_prob:
#         disease = "WSSV"
#         st.warning("⚠️ White Spot Syndrome Virus Detected")
#         recommendation = "Isolate affected prawns immediately."
#     else:
#         disease = "BG"
#         st.success("✅ Black Gill Disease / Healthy")
#         recommendation = "Maintain water quality and routine monitoring."

#     st.markdown("### 📊 Confidence Scores")
#     st.progress(bg_prob)
#     st.write(f"BG Probability: {bg_prob * 100:.2f}%")

#     st.progress(wssv_prob)
#     st.write(f"WSSV Probability: {wssv_prob * 100:.2f}%")

#     st.markdown("### 🧠 Recommendation")
#     st.info(recommendation)

# else:
#     st.info("⬆️ Upload or scan an image to start detection")
# confidence_threshold = st.slider(
#     "🎯 Confidence Threshold",
#     min_value=0.3,
#     max_value=0.9,
#     value=0.6,
#     step=0.05,
#     help="Minimum confidence required to accept a prediction"
# )








































import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Prawn Disease Detection",
    page_icon="🦐",
    layout="centered"
)

st.title("🦐 Prawn Disease Detection App")
st.write("Upload or scan a prawn image to detect diseases")

# --------------------------------------------------
# MODEL PATH
# --------------------------------------------------
MODEL_PATH = "model/best_model.pt"

# --------------------------------------------------
# LOAD MODEL (MATCHES TRAINING)
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

model = load_model()
st.success("✅ Classification model loaded successfully")

# --------------------------------------------------
# IMAGE TRANSFORM (SAME AS TRAINING)
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# CONFIDENCE THRESHOLD
# --------------------------------------------------
confidence_threshold = st.slider(
    "🎯 Confidence Threshold",
    min_value=0.3,
    max_value=0.9,
    value=0.6,
    step=0.05,
    help="Minimum confidence required to accept a disease prediction"
)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
st.subheader("📤 Upload or Scan Image")

uploaded_file = st.file_uploader(
    "Upload prawn image",
    type=["jpg", "jpeg", "png"]
)

camera_image = st.camera_input("📷 Scan using camera")

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif camera_image is not None:
    image = Image.open(camera_image).convert("RGB")

# --------------------------------------------------
# PREDICTION LOGIC
# --------------------------------------------------
if image is not None:
    st.image(image, caption="Input Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs)[0]

    bg_prob = probs[0].item()
    wssv_prob = probs[1].item()

    # -------------------------
    # DISEASE DECISION LOGIC
    # -------------------------
    if bg_prob > confidence_threshold and wssv_prob > confidence_threshold:
        disease = "WSSV_BG"
        st.error("🚨 Mixed Infection Detected (BG + WSSV)")
        recommendation = "Immediate expert intervention required."

    elif wssv_prob > confidence_threshold:
        disease = "WSSV"
        st.warning("⚠️ White Spot Syndrome Virus (WSSV) Detected")
        recommendation = "Isolate affected prawns immediately."

    elif bg_prob > confidence_threshold:
        disease = "BG"
        st.warning("⚠️ Black Gill Disease (BG) Detected")
        recommendation = "Improve water quality and monitor closely."

    else:
        disease = "HEALTHY"
        st.success("✅ Healthy Prawn Detected")
        recommendation = "No disease detected. Continue routine pond management."

    # --------------------------------------------------
    # CONFIDENCE DISPLAY
    # --------------------------------------------------
    st.markdown("### 📊 Confidence Scores")

    st.write(f"BG Probability: {bg_prob * 100:.2f}%")
    st.progress(bg_prob)

    st.write(f"WSSV Probability: {wssv_prob * 100:.2f}%")
    st.progress(wssv_prob)

    # --------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------
    st.markdown("### 🧠 Interpretation")
    if disease == "HEALTHY":
        st.caption(
            "Both disease probabilities are below the confidence threshold, "
            "so the prawn is considered healthy."
        )

    # --------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------
    st.markdown("### 🩺 Recommendation")
    st.info(recommendation)

else:
    st.info("⬆️ Upload or scan an image to start detection")
