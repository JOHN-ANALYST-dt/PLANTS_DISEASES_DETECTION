# disease_interventions.py

# Dictionary containing detailed intervention strategies for specific crop diseases.
# Keys MUST match the exact class names used in your FULL_CLASS_NAMES list 
# (although the get_interventions function includes fallback logic).

INTERVENTIONS = {
    # -------------------------- HEALTHY STATUS --------------------------
    "Healthy": {
        "title": "🎉 Status: HEALTHY!",
        "action": [
            "**Farm Insight:** Your crop is currently thriving and free from these common diseases. Keep up the excellent work!",
            "**Recommended Action:** Maintain optimal growing conditions: consistent watering, balanced nutrients, and good air circulation.",
            "**Prevention Tip:** Regularly scout your fields (inspecting both the top and underside of leaves) to catch any early signs of pests or disease immediately."
        ]
    },
    "cabbage healthy": {
        "title": "🎉 Status: Cabbage is HEALTHY!",
        "action": [
            "**Farm Insight:** Cabbage plants show no signs of clubroot or black rot. Continue monitoring.",
            "**Recommended Action:** Ensure soil pH is maintained (7.0-7.2) to naturally suppress clubroot.",
            "**Prevention Tip:** Practice crop rotation (3-4 years) with non-brassica crops."
        ]
    },
    "potato healthy": {
        "title": "🎉 Status: Potato is HEALTHY!",
        "action": [
            "**Farm Insight:** No signs of early or late blight detected. The canopy looks strong.",
            "**Recommended Action:** Hill your potato rows to protect developing tubers from light and pests.",
            "**Prevention Tip:** Use certified disease-free seed potatoes for planting."
        ]
    },
    "Tomato Healthy": {
        "title": "🎉 Status: Tomato is HEALTHY!",
        "action": [
            "**Farm Insight:** Your tomatoes are free from bacterial and viral threats. Encourage vigorous growth.",
            "**Recommended Action:** Prune suckers and lower leaves to improve air flow and prevent ground splash.",
            "**Prevention Tip:** Apply a layer of mulch (straw or plastic) to minimize soil contact with the lower leaves."
        ]
    },
    "Pepper Bell Healthy": {
        "title": "🎉 Status: Pepper is HEALTHY!",
        "action": [
            "**Farm Insight:** Excellent growth. The pepper plant shows no signs of bacterial spot.",
            "**Recommended Action:** Ensure consistent moisture to prevent blossom-end rot.",
            "**Prevention Tip:** Space plants adequately and use drip irrigation to keep foliage dry."
        ]
    },
    "Grape Healthy": {
        "title": "🎉 Status: Grape is HEALTHY!",
        "action": [
            "**Farm Insight:** The vine and foliage appear strong and ready for fruit production.",
            "**Recommended Action:** Maintain a well-managed trellis system to maximize sun exposure and air circulation.",
            "**Prevention Tip:** Prune to an open canopy structure before spring to reduce fungal risk."
        ]
    },
    "Cherry Healthy": {
        "title": "🎉 Status: Cherry is HEALTHY!",
        "action": [
            "**Farm Insight:** No signs of powdery mildew. The tree is in good condition.",
            "**Recommended Action:** Monitor soil for adequate drainage, as wet feet stress the tree.",
            "**Prevention Tip:** Prune dormant wood to increase light penetration."
        ]
    },
    "Strawberry Healthy": {
        "title": "🎉 Status: Strawberry is HEALTHY!",
        "action": [
            "**Farm Insight:** Leaves are free of scorch and fungal spots. Focus on berry quality.",
            "**Recommended Action:** Renew the mulch layer (straw) to keep berries clean and suppress weeds.",
            "**Prevention Tip:** Renovate matted rows immediately after harvest to maintain plant vigor."
        ]
    },
    "skumawiki healthy": {
        "title": "🎉 Status: Skumawiki is HEALTHY!",
        "action": ["Maintain current care practices.", "Ensure adequate water during dry periods."]
    },
    "soybean healthy": {
        "title": "🎉 Status: Soybean is HEALTHY!",
        "action": ["Ensure adequate soil nutrition, particularly potassium.", "Scout for insect pests, which can transmit viruses."]
    },
    "tobacco healthy leaf": {
        "title": "🎉 Status: Tobacco is HEALTHY!",
        "action": ["Avoid handling plants when wet to prevent bacterial spread.", "Ensure proper ventilation in curing barns."]
    },
    "raspberry healthy": {
        "title": "🎉 Status: Raspberry is HEALTHY!",
        "action": ["Maintain rows at a manageable width.", "Ensure good trellising for canes to improve air flow."]
    },
    "peach healthy": {
        "title": "🎉 Status: Peach is HEALTHY!",
        "action": ["Apply dormant oil in late winter to manage overwintering pests.", "Thin fruit to prevent branch breakage and improve size."]
    },
    "onion healthy leaf": {
        "title": "🎉 Status: Onion is HEALTHY!",
        "action": ["Ensure excellent soil drainage.", "Use drip irrigation to keep the bulb neck dry."]
    },
    
    # -------------------------- APPLE DISEASES --------------------------
    "Apple Scab": {
        "title": "🔴 Urgent Action: Apple Scab (Fungal)",
        "action": [
            "**Pruning/Sanitation:** Rake up and destroy all infected fallen leaves in the autumn to eliminate the primary source of infection.",
            "**Chemical Control:** Apply a systemic fungicide (e.g., myclobutanil) starting at bud break, repeating as directed, especially during wet periods.",
            "**Farm Advice:** Prioritize air flow. Prune your trees annually to open the canopy."
        ]
    },
    "Apple Black Rot": {
        "title": "🔴 Urgent Action: Apple Black Rot (Fungal)",
        "action": [
            "**Physical Removal:** Remove and destroy all **mummified fruit** (shriveled, dead fruit) from the tree and ground immediately.",
            "**Pruning:** Cut out and destroy any dead wood or cankers on branches, cutting several inches below the infected tissue.",
            "**Chemical Control:** Use a recommended fungicide (e.g., Captan) starting at the pink bud stage and continuing through the cover sprays."
        ]
    },
    "Apple Cedar Rust": {
        "title": "🔴 Urgent Action: Apple Cedar Rust (Fungal)",
        "action": [
            "**Host Removal:** If possible, remove wild cedar (Juniperus species) trees within a few hundred yards of the orchard, as they are the alternate host.",
            "**Chemical Control:** Apply fungicides (e.g., containing myclobutanil or propiconazole) from the pink bud stage through late bloom, focusing on new foliage.",
            "**Farm Advice:** Choose resistant apple cultivars for future plantings."
        ]
    },

    # -------------------------- CABBAGE DISEASES --------------------------
    "cabbage black rot": {
        "title": "🔴 Urgent Action: Cabbage Black Rot (Bacterial)",
        "action": [
            "**Sanitation:** Immediately remove and destroy infected plants. Do NOT compost them.",
            "**Cultural:** Practice strict crop rotation (3-5 years) with non-brassica crops.",
            "**Chemical Control:** Copper-based products can offer some suppression if applied early, but sanitation and prevention are key. Avoid overhead watering."
        ]
    },
    "cabbage clubroot": {
        "title": "🔴 Severe Action: Cabbage Clubroot (Slime Mold)",
        "action": [
            "**Soil Amendment:** Clubroot thrives in acidic soil. Adjust soil pH to 7.2 or higher by adding lime (calcium hydroxide).",
            "**Cultural:** Avoid moving soil from infested areas to healthy areas. Clean tools thoroughly.",
            "**Farm Advice:** Once established, clubroot is difficult to eradicate. Plant only resistant varieties in known infested areas."
        ]
    },
    "cabbage downy mildew": {
        "title": "🔴 Action: Cabbage Downy Mildew (Oomycete)",
        "action": [
            "**Air Circulation:** Increase plant spacing and use drip irrigation to reduce leaf wetness and humidity.",
            "**Chemical Control:** Apply an approved fungicide specifically for downy mildew, ensuring full coverage of the underside of the leaves."
        ]
    },
    "cabbage leaf disease": {
        "title": "🟡 General Action: Cabbage Leaf Disease",
        "action": [
            "Inspect closely to determine if the cause is fungal, bacterial, or environmental.",
            "Remove and discard severely affected outer leaves.",
            "Apply a broad-spectrum fungicide or bactericide (copper) depending on observed symptoms."
        ]
    },

    # -------------------------- CORN DISEASES --------------------------
    "Corn Common Rust": {
        "title": "🟡 Action: Corn Common Rust (Fungal)",
        "action": [
            "**Chemical Control:** Fungicide applications (e.g., triazoles or strobilurins) are recommended if the disease develops early and severely before the tasseling stage.",
            "**Farm Advice:** Plant resistant hybrids next season, as this is the most effective long-term control."
        ]
    },
    "Corn Northern leaf blight": {
        "title": "🔴 Urgent Action: Corn Northern Leaf Blight (Fungal)",
        "action": [
            "**Chemical Control:** Fungicide application is crucial if lesions are seen on the upper leaves before or near the silking stage.",
            "**Cultural:** Till under infected crop residue after harvest to reduce overwintering spores."
        ]
    },
    "Corn Cercospora Leaf Spot gray leaf spot": {
        "title": "🔴 Urgent Action: Corn Gray Leaf Spot (Fungal)",
        "action": [
            "**Chemical Control:** Fungicides are necessary, particularly in susceptible corn hybrids, starting before the disease becomes widespread.",
            "**Cultural:** Rotate crops with non-host crops like soybeans or alfalfa to break the disease cycle."
        ]
    },

    # -------------------------- POTATO DISEASES --------------------------
    "Potato Early Blight": {
        "title": "🔴 Action: Potato Early Blight (Fungal)",
        "action": [
            "**Chemical Control:** Initiate a fungicide spray program early, before visible symptoms, especially in susceptible varieties.",
            "**Cultural:** Avoid nutrient stress, particularly low nitrogen, by maintaining a steady fertilization program."
        ]
    },
    "Potato Late Blight": {
        "title": "🔥 Immediate Action: Potato Late Blight (Oomycete)",
        "action": [
            "**Chemical Control:** Immediate and aggressive use of specific late blight fungicides is mandatory to stop spread.",
            "**Physical Control:** Remove and destroy infected foliage to prevent spores from reaching the tubers.",
            "**Farm Advice:** This disease is highly destructive; constant vigilance and timely spraying are essential in humid weather."
        ]
    },

    # -------------------------- TOMATO DISEASES --------------------------
    "Tomato Bacterial Spot": {
        "title": "🔴 Action: Tomato Bacterial Spot (Bacterial)",
        "action": [
            "**Chemical Control:** Apply copper-based bactericides combined with mancozeb at weekly intervals.",
            "**Cultural:** Avoid overhead irrigation. Do not touch or work on plants when foliage is wet, as this spreads the bacteria."
        ]
    },
    "Tomato Early Blight": {
        "title": "🔴 Action: Tomato Early Blight (Fungal)",
        "action": [
            "**Chemical Control:** Apply protective fungicides (e.g., containing chlorothalonil) starting when the first fruit clusters set.",
            "**Pruning:** Remove the lowest, oldest, and most infected leaves to reduce the spore source near the soil."
        ]
    },
    "Tomato late blight": {
        "title": "🔥 Immediate Action: Tomato Late Blight (Oomycete)",
        "action": [
            "**Chemical Control:** Requires immediate application of specialized fungicides to halt the rapid spread.",
            "**Sanitation:** Destroy infected plants immediately to protect remaining healthy plants."
        ]
    },
    "Tomato leaf mold": {
        "title": "🟡 Action: Tomato Leaf Mold (Fungal)",
        "action": [
            "**Air Circulation:** Improve ventilation around plants by pruning and increasing spacing.",
            "**Cultural:** If growing in a greenhouse, reduce humidity and avoid condensation on leaves."
        ]
    },
    "Tomato septoria leaf spot": {
        "title": "🔴 Action: Tomato Septoria Leaf Spot (Fungal)",
        "action": [
            "**Chemical Control:** Begin fungicide treatment (chlorothalonil) at the first sign of spotting.",
            "**Cultural:** Remove infected leaves and practice crop rotation, as spores survive on debris."
        ]
    },
    "Tomato spider mites Two-spotted spider mite": {
        "title": "🔴 Action: Two-spotted Spider Mite (Pest)",
        "action": [
            "**Physical Control:** Use a strong jet of water to physically wash mites and their webs off the leaves, focusing on the undersides.",
            "**Chemical/Bio Control:** Apply horticultural oil or an approved miticide. In enclosed spaces, release beneficial predatory mites."
        ]
    },
    "Tomato Target Spot": {
        "title": "🔴 Action: Tomato Target Spot (Fungal)",
        "action": [
            "**Chemical Control:** Apply protective fungicides regularly, focusing on coverage of all foliage.",
            "**Cultural:** Stake plants to keep them off the ground and use mulch to prevent soil splash."
        ]
    },
    "Tomato Yellow Leaf Curl Virus": {
        "title": "🚨 NO CURE: Tomato Yellow Leaf Curl Virus (Viral)",
        "action": [
            "**Quarantine:** Immediately remove and destroy all infected plants. **There is no chemical cure for the virus itself.**",
            "**Vector Control:** Control the **Whitefly** (the virus carrier) by applying insecticides or reflective mulch to repel them.",
            "**Farm Advice:** For next season, use only resistant tomato varieties."
        ]
    },
    "Tomato mosaic virus": {
        "title": "🚨 NO CURE: Tomato Mosaic Virus (Viral)",
        "action": [
            "**Quarantine:** Immediately remove and destroy all infected plants to prevent transmission.",
            "**Sanitation:** Wash hands, tools, and surfaces with a 10% bleach solution or soap after handling infected plants, as the virus is easily spread by touch.",
            "**Prevention:** Avoid using any tobacco products while working with plants."
        ]
    },

    # -------------------------- GRAPE DISEASES --------------------------
    "Grape Black Rot": {
        "title": "🔴 Urgent Action: Grape Black Rot (Fungal)",
        "action": [
            "**Sanitation:** Remove all infected mummies (shriveled grapes) from the vine and ground during winter pruning.",
            "**Chemical Control:** Begin fungicide applications early in the season, typically when new shoots are 1-3 inches long, and continue until veraison (color change)."
        ]
    },
    "Grape Esca (Black Measles)": {
        "title": "🔴 Long-Term Management: Grape Esca (Fungal)",
        "action": [
            "**Pruning:** Prune heavily infected cordons (arms) and trunks back to healthy wood.",
            "**Long-term:** Avoid large pruning wounds when possible, and apply wound protectants after pruning."
        ]
    },
    "Grape Leaf Blight (Isariopsis Leaf Spot)": {
        "title": "🟡 Action: Grape Leaf Blight (Fungal)",
        "action": [
            "**Chemical Control:** Apply fungicides, often in combination with other treatments, especially in humid conditions.",
            "**Cultural:** Ensure good air circulation through proper canopy management."
        ]
    },
    
    # -------------------------- OTHER FRUITS --------------------------
    "Cherry Powdery Mildew": {
        "title": "🟡 Action: Cherry Powdery Mildew (Fungal)",
        "action": [
            "**Chemical Control:** Use approved fungicides, sulfur, or horticultural oils (avoid sulfur if oils are used) starting at shuck fall.",
            "**Cultural:** Prune to improve light penetration and air movement within the tree canopy."
        ]
    },
    "Strawberry Leaf Scorch": {
        "title": "🟡 Action: Strawberry Leaf Scorch (Fungal)",
        "action": [
            "**Cultural:** Use renovation practices (mowing and thinning) after harvest to remove infected foliage.",
            "**Chemical Control:** Apply fungicides at appropriate growth stages (e.g., pre-bloom and post-harvest)."
        ]
    },
    "soybean frog eye leaf spot": {
        "title": "🔴 Action: Soybean Frog-eye Leaf Spot (Fungal)",
        "action": [
            "**Chemical Control:** Fungicide application is recommended if disease develops early, particularly if planting is done using susceptible varieties.",
            "**Cultural:** Rotate crops to reduce residue inoculum."
        ]
    },
    "soybean rust": {
        "title": "🔥 Immediate Action: Soybean Rust (Fungal)",
        "action": [
            "**Chemical Control:** Timely fungicide application is critical, often needing to be preventative or applied immediately upon detection.",
            "**Farm Advice:** Monitor regional warnings for spore migration, as this is a wind-blown disease."
        ]
    },
    "soybean powdery mildew": {
        "title": "🟡 Action: Soybean Powdery Mildew (Fungal)",
        "action": [
            "**Chemical Control:** Typically manageable, but fungicides may be needed late in the season if infection is severe.",
            "**Cultural:** Ensure adequate plant spacing for air movement."
        ]
    },
    "tobacco black shank": {
        "title": "🔥 Severe Action: Tobacco Black Shank (Oomycete)",
        "action": [
            "**Sanitation:** Immediately remove infected plants.",
            "**Cultural:** Plant resistant varieties and maintain proper drainage, as wet soils favor the disease.",
            "**Chemical Control:** Apply fungicides (e.g., mefenoxam) in transplant water or soil, followed by lay-by applications."
        ]
    },
    "tobacco leaf disease": {
        "title": "🟡 General Action: Tobacco Leaf Disease",
        "action": [
            "Identify if the cause is bacterial (use copper) or fungal (use fungicide).",
            "Monitor soil moisture and humidity."
        ]
    },
    "tobacco mosaic virus": {
        "title": "🚨 NO CURE: Tobacco Mosaic Virus (Viral)",
        "action": [
            "**Sanitation:** Immediately remove infected plants. Clean hands and tools thoroughly.",
            "**Prevention:** Do not use any tobacco products (cigarettes, cigars) while working in the field, as the virus is easily transmitted."
        ]
    },
    "raspberry leaf spot": {
        "title": "🟡 Action: Raspberry Leaf Spot (Fungal)",
        "action": [
            "**Pruning:** Remove old, fruited canes (floricanes) immediately after harvest.",
            "**Chemical Control:** Apply fungicides during the early spring before bloom."
        ]
    },
    "peach bacterial spot": {
        "title": "🔴 Action: Peach Bacterial Spot (Bacterial)",
        "action": [
            "**Cultural:** Minimize tree stress. Do not work in the orchard when foliage is wet.",
            "**Chemical Control:** Apply copper-based sprays during the dormant and early growth stages."
        ]
    },
    "peach leaf curl": {
        "title": "🔴 Action: Peach Leaf Curl (Fungal)",
        "action": [
            "**Chemical Control:** Apply a single dormant fungicide spray (copper or chlorothalonil) in **late autumn** after leaf fall or in **early spring** before bud swell. **Crucially: Sprays are ineffective once leaves show symptoms.**"
        ]
    },
    "peach powdery mildew": {
        "title": "🟡 Action: Peach Powdery Mildew (Fungal)",
        "action": [
            "**Chemical Control:** Apply sulfur or other approved fungicides, starting at the shuck-split stage.",
            "**Cultural:** Ensure good air circulation through pruning."
        ]
    },
    "peach leaf disease": {
        "title": "🟡 General Action: Peach Leaf Disease",
        "action": [
            "Determine if the issue is fungal (most likely) or pest-related.",
            "Apply a broad-spectrum fungicide and ensure good orchard hygiene."
        ]
    },
    "orange citrus greening": {
        "title": "🚨 CATASTROPHIC: Citrus Greening (Bacterial)",
        "action": [
            "**NO CURE:** This disease is terminal and incurable. **Immediately remove and destroy the infected tree** to protect the rest of the grove.",
            "**Vector Control:** Aggressively control the Asian Citrus Psyllid (the insect vector) with targeted insecticides.",
            "**Farm Advice:** Use only certified disease-free nursery stock for new plantings."
        ]
    },
    "orange leaf curl": {
        "title": "🟡 Action: Orange Leaf Curl (Pest/Environmental)",
        "action": [
            "Check for aphids or other sucking insects, which cause distortion. Control the pests.",
            "Ensure the tree is not experiencing severe water stress, which can also cause leaf curl."
        ]
    },
    "orange leaf disease": {
        "title": "🟡 General Action: Orange Leaf Disease",
        "action": [
            "Identify specific symptoms (e.g., oily spots, chlorosis) and consult citrus-specific guides.",
            "Apply copper fungicides which are standard for many citrus fungal/bacterial diseases."
        ]
    },
    "orange leaf spot": {
        "title": "🟡 Action: Orange Leaf Spot (Fungal/Bacterial)",
        "action": [
            "**Chemical Control:** Apply a copper spray if the spot is confirmed bacterial or fungal.",
            "**Cultural:** Prune to increase light and air penetration, promoting dry leaves."
        ]
    },
    "onion downy mildew": {
        "title": "🔥 Immediate Action: Onion Downy Mildew (Oomycete)",
        "action": [
            "**Chemical Control:** Immediate application of a highly effective systemic fungicide is required.",
            "**Cultural:** Stop all overhead irrigation. Reduce humidity and dew periods as much as possible."
        ]
    },
    "onion leaf blight": {
        "title": "🔴 Action: Onion Leaf Blight (Fungal)",
        "action": [
            "**Chemical Control:** Initiate fungicide application at the first sign of symptoms and continue at regular intervals.",
            "**Cultural:** Ensure proper nitrogen-to-potassium balance, as excess nitrogen can increase susceptibility."
        ]
    },
    "onion purple blotch": {
        "title": "🔴 Action: Onion Purple Blotch (Fungal)",
        "action": [
            "**Chemical Control:** Use protectant fungicides, often starting early and continuing through bulb development.",
            "**Cultural:** Destroy infested cull piles and volunteer onions, which can harbor the fungus."
        ]
    },
    "onion thrips damage": {
        "title": "🔴 Action: Onion Thrips Damage (Pest)",
        "action": [
            "**Pest Control:** Use an approved insecticide targeted at thrips. Target the leaf axils where they hide.",
            "**Cultural:** Use high-pressure irrigation sprays to knock thrips off the plants.",
            "**Farm Advice:** Thrips damage appears as silvery streaks. Catching them early is key to preventing yield loss."
        ]
    },
    
    "skumawiki leaf disease": {
        "title": "🟡 General Action: Unknown Leaf Disease",
        "action": [
            "As the exact disease is unknown, apply a broad-spectrum control: one application of a copper-based bactericide and one application of a broad-spectrum fungicide.",
            "Immediately remove and destroy severely affected leaves and plants."
        ]
    },
    "default": {
        "title": "⚠️ General Advice (Diagnosis Uncertain)",
        "action": [
            "**Quarantine:** Isolate the plant or area to prevent possible spread while awaiting a definitive diagnosis.",
            "**Consultation:** Take high-resolution photos and consult a local agricultural expert or extension office.",
            "**Interim Action:** Maintain excellent hygiene: clean all tools, hands, and equipment after working near the plant."
        ]
    }
}

# LOOKUP FUNCTION 

def get_interventions(predicted_class):
    """Retrieves specific interventions based on the predicted class name."""
    
    # Check for exact match first (which is the most reliable method)
    if predicted_class in INTERVENTIONS:
        return INTERVENTIONS[predicted_class]

    # Handle slight variations in 'healthy' names (e.g., 'Tomato Healthy' vs 'Healthy')
    if "healthy" in predicted_class.lower():
        return INTERVENTIONS.get("Healthy")

    # Fallback for general categories
    if "leaf disease" in predicted_class.lower() or "leaf blight" in predicted_class.lower():
        # A more specific general intervention can be created if needed, 
        # but for now, we rely on the specific keys above.
        return INTERVENTIONS.get("skumawiki leaf disease")
        
    # Return default intervention if nothing is found
    return INTERVENTIONS.get("default")