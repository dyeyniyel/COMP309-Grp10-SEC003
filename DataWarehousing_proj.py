import pandas as pd
import numpy as np
import joblib
import gradio as gr

# Load CSV containing gender probabilities for each nature of contact
prob_df = pd.read_csv("gender_prob_conduct.csv")

# ---- CONSTANTS / CHOICES ----
# Columns used by the trained Random Forest model
MODEL_COLUMNS = [
    'AGE', 'Month_birth', 'MALE_PROB_NATURE', 'FEMALE_PROB_NATURE',
    'NATURE_OF_CONTACT_CPIC 10-66 Return to Investigator',
    'NATURE_OF_CONTACT_Dispute (Non-Domestic)',
    'NATURE_OF_CONTACT_Drug Related',
    'NATURE_OF_CONTACT_EDP Related',
    'NATURE_OF_CONTACT_General Info for Intelligence Unit',
    'NATURE_OF_CONTACT_General Investigation',
    'NATURE_OF_CONTACT_Liquor Licence Act',
    'NATURE_OF_CONTACT_Loitering',
    'NATURE_OF_CONTACT_Radio Call',
    'NATURE_OF_CONTACT_Sex Trade Related',
    'NATURE_OF_CONTACT_Shoplifting',
    'NATURE_OF_CONTACT_Squeegee Kid/Panhandler/Strt Person',
    'NATURE_OF_CONTACT_Suspicious Activity',
    'NATURE_OF_CONTACT_Traffic Stop',
    'NATURE_OF_CONTACT_Trespassing',
    'NATURE_OF_CONTACT_Vehicle Related',
    'SEASON_Spring', 'SEASON_Summer', 'SEASON_Winter',
    'SKIN_COLOUR_Brown', 'SKIN_COLOUR_White'
]

# Dropdown choices for "Nature of contact"
NATURE_OF_CONTACT_CHOICES = [
    'Traffic Stop',
    'General Investigation',
    'General Info for Intelligence Unit',
    'Vehicle Related',
    'EDP Related',
    'CPIC 10-66 Return to Investigator',
    'Shoplifting',
    'Liquor Licence Act',
    'Suspicious Activity',
    'Drug Related',
    'Trespassing',
    'Squeegee Kid/Panhandler/Strt Person',
    'Sex Trade Related',
    'Loitering',
    'Bail Compliance Check-No Violation',
    'Dispute (Non-Domestic)',
    'Radio Call',
]

# Dropdown choices for season, skin colour, and month of birth
SEASON_CHOICES = ["Spring", "Summer", "Fall", "Winter"]
SKIN_COLOUR_CHOICES = ["Brown", "White", "Black"]
MONTH_BIRTH_CHOICES = list(range(1, 13))  # 1–12

# ---- FUNCTIONS ----
def lookup_probs_by_nature(nature: str):
    """
    Look up precomputed male/female probabilities for a given nature of contact.
    Falls back to 0.5 / 0.5 if the nature is not found in the CSV.
    """
    row = prob_df.loc[prob_df["NATURE_OF_CONTACT"] == nature]
    if row.empty:
        return 0.5, 0.5
    male_prob = float(row["MALE_PROB_NATURE"].iloc[0])
    female_prob = float(row["FEMALE_PROB_NATURE"].iloc[0])
    return male_prob, female_prob

def preprocess_single(age, month_birth, nature, season, skin_colour):
    """
    Build a single-row feature matrix from raw user inputs.
    This matches the training-time feature engineering (dummies + numeric).
    """
    # Get gender probabilities for this nature of contact
    male_prob, female_prob = lookup_probs_by_nature(nature)

    # Raw input row
    raw = pd.DataFrame([{
        'AGE': age,
        'Month_birth': month_birth,
        'MALE_PROB_NATURE': male_prob,
        'FEMALE_PROB_NATURE': female_prob,
        'NATURE_OF_CONTACT': nature,
        'SEASON': season,
        'SKIN_COLOUR': skin_colour
    }])

    # One-hot encode categorical variables
    dummies = pd.get_dummies(
        raw,
        columns=['NATURE_OF_CONTACT', 'SEASON', 'SKIN_COLOUR'],
        drop_first=False,
        dtype=int
    )

    # Start with all-zero columns, then fill what we have
    X = pd.DataFrame(
        np.zeros((1, len(MODEL_COLUMNS))),
        columns=MODEL_COLUMNS
    )

    # Fill one-hot columns that exist in the model
    for col in dummies.columns:
        if col in X.columns:
            X[col] = dummies[col].values

    # Copy over numeric / base columns
    for base in ['AGE', 'Month_birth', 'MALE_PROB_NATURE', 'FEMALE_PROB_NATURE']:
        if base in raw.columns and base in X.columns:
            X[base] = raw[base].values

    return X

# ---- MODEL & PREDICT ----
# Load the trained Random Forest gender model
model = joblib.load("random_forest_gender_model.pkl")

def predict_fn(age, month_birth, nature, season, skin_colour):
    """
    Gradio callback: takes user inputs, preprocesses them,
    runs the model, and returns male/female percentages.
    """
    X = preprocess_single(age, month_birth, nature, season, skin_colour)

    # Model outputs probability of "male" class
    male_proba = float(model.predict_proba(X)[0, 1])
    female_proba = float(1.0 - male_proba)

    # Convert to percentages with 2 decimal places
    male_pct = round(male_proba * 100, 2)
    female_pct = round(female_proba * 100, 2)
    return male_pct, female_pct

# Optional test call (AFTER everything is defined)
# print(predict_fn(21, 3, "Shoplifting", "Fall", "Brown"))

# ---- GRADIO UI ----
# Custom CSS to style the Gradio app (background, cards, etc.)
custom_css = """
/* Background: dark gradient + subtle pattern */
body {
    background:
        radial-gradient(circle at top, rgba(56, 189, 248, 0.16), transparent 55%),
        radial-gradient(circle at bottom, rgba(37, 99, 235, 0.22), #020617);
    background-attachment: fixed;
    color: #e5e7eb;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* subtle grid overlay */
body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: 0.18;
    background-image:
        linear-gradient(120deg, rgba(148, 163, 184, 0.18) 1px, transparent 1px),
        linear-gradient(210deg, rgba(30, 64, 175, 0.16) 1px, transparent 1px);
    background-size: 110px 110px, 140px 140px;
    mix-blend-mode: soft-light;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 40px auto !important;
    background: transparent !important;
}

/* Glassy main card */
.card {
    background: rgba(15, 23, 42, 0.78);
    padding: 28px;
    border-radius: 22px;
    border: 1px solid rgba(129, 140, 248, 0.6);
    box-shadow:
        0 24px 60px rgba(15, 23, 42, 0.95),
        0 0 40px rgba(37, 99, 235, 0.45);
}

/* Header with police icon */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 8px;
    justify-content: center; /* center the whole header block */
}

.app-icon {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 20%, #f97373, #0ea5e9);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 20px rgba(248, 113, 113, 0.7);
    font-size: 1.4rem;
}

.app-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #e5edff;
    text-align: center; /* ensure text is centered */
}

.app-subtitle {
    color: #9ca3af;
    margin-bottom: 1.4rem;
    text-align: center;
}

/* Section title */
.section-title {
    font-weight: 600;
    color: #93c5fd;
    margin: 12px 0 6px 0;
}

/* Labels */
label, .wrap.svelte-1ipelgc, .wrap.svelte-1g805jl {
    color: #e5e7eb !important;
}

/* Inputs / dropdowns */
input, select, textarea {
    background: rgba(15, 23, 42, 0.7) !important;
    color: #f9fafb !important;
    border-radius: 10px !important;
    border: 1px solid rgba(148, 163, 184, 0.6) !important;
    padding: 10px !important;
    transition:
        background 0.2s ease,
        border-color 0.2s ease,
        box-shadow 0.2s ease;
}

/* Hover */
input:hover, select:hover, textarea:hover {
    background: rgba(30, 64, 118, 0.9) !important;
    border-color: #60a5fa !important;
}

/* Focus */
input:focus, select:focus, textarea:focus {
    background: rgba(15, 23, 42, 0.95) !important;
    border-color: #3b82f6 !important;
    outline: none !important;
    box-shadow: 0 0 12px rgba(59, 130, 246, 0.7) !important;
}

/* Predict button */
.gr-button {
    background: linear-gradient(90deg, #2563eb, #38bdf8) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 999px !important;
    padding: 12px 20px !important;
    border: none !important;
    box-shadow: 0 10px 25px rgba(37, 99, 235, 0.7);
    transition:
        box-shadow 0.15s ease-out,
        filter 0.15s ease-out;
}

.gr-button:hover {
    filter: brightness(1.05);
    box-shadow: 0 14px 32px rgba(37, 99, 235, 0.9);
}

/* Right side output panel */
.output-panel {
    padding-left: 16px;
    border-left: 1px solid rgba(51, 65, 85, 0.9);
}

/* Output cards */
.output-card {
    background: rgba(15, 23, 42, 0.86);
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.55);
    padding: 16px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.85);
    margin-bottom: 12px;
}

.output-card input {
    background: rgba(15, 23, 42, 0.98) !important;
    color: #f8fafc !important;
    border-color: rgba(148, 163, 184, 0.5) !important;
}

/* Make Age and Month of Birth fields equal width */
.uniform-input input,
.uniform-input select {
    width: 100% !important;
    min-width: 0 !important;
}

/* Custom footer text centering */
#custom-footer {
    text-align: center;
    margin-top: 16px;
    color: #e5e7eb;
}
"""

with gr.Blocks() as demo:
    # Inject custom CSS into the Gradio app
    gr.HTML(f"<style>{custom_css}</style>")

    with gr.Column(elem_classes="card"):
        # ----- Header (title + subtitle) -----
        gr.HTML(
            """
            <div class="app-header">
                <div class="app-icon">🚔</div>
                <div>
                    <div class="app-title">Nature of Contact Risk Model</div>
                    <div class="app-subtitle">
                        Explore how age, nature of contact, season, and skin colour relate to predicted gender probabilities.
                    </div>
                </div>
            </div>
            """
        )

        # Two-column layout: left = inputs, right = prediction outputs
        with gr.Row():
            # LEFT: Input controls
            with gr.Column(scale=6):
                gr.Markdown("### Inputs", elem_classes="section-title")

                with gr.Row():
                    # Numeric age input
                    with gr.Column(scale=1, elem_classes="uniform-input"):
                        age_in = gr.Number(label="Age", precision=0)

                    # Month of birth dropdown
                    with gr.Column(scale=1, elem_classes="uniform-input"):
                        month_birth_in = gr.Dropdown(
                            label="Month of Birth (1–12)",
                            choices=MONTH_BIRTH_CHOICES
                        )

                # Other dropdown inputs
                nature_in = gr.Dropdown(
                    label="Nature of contact",
                    choices=NATURE_OF_CONTACT_CHOICES
                )
                season_in = gr.Dropdown(
                    label="Season",
                    choices=SEASON_CHOICES
                )
                skin_in = gr.Dropdown(
                    label="Skin Colour",
                    choices=SKIN_COLOUR_CHOICES
                )

                # Button to trigger prediction
                submit_btn = gr.Button("Predict")

            # RIGHT: Prediction results
            with gr.Column(scale=4, elem_classes="output-panel"):
                gr.Markdown("### Prediction Results", elem_classes="section-title")

                # Predicted male percentage
                with gr.Column(elem_classes="output-card"):
                    male_out = gr.Number(
                        label="Predicted male probability (%)",
                        precision=2
                    )

                # Predicted female percentage (complement)
                with gr.Column(elem_classes="output-card"):
                    female_out = gr.Number(
                        label="Predicted female probability (%) (100 - male)",
                        precision=2
                    )

        # Wire button click to prediction function
        submit_btn.click(
            fn=predict_fn,
            inputs=[age_in, month_birth_in, nature_in, season_in, skin_in],
            outputs=[male_out, female_out]
        )

        # Centered custom footer text below the app
        gr.Markdown(
            "COMP309 - Group10 - SEC003",
            elem_id="custom-footer"
        )

if __name__ == "__main__":
    # Launch Gradio interface
    demo.launch()
