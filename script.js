import {
    pipeline,
    env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";


// =====================================================
// TRANSFORMERS.JS SETTINGS
// =====================================================

env.allowLocalModels = false;
env.useBrowserCache = true;


// =====================================================
// GLOBAL VARIABLES
// =====================================================

let safetyClassifier = null;
let selectedImage = null;
let modelLoading = false;


// =====================================================
// GET HTML ELEMENTS
// =====================================================

const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const previewContainer = document.getElementById("previewContainer");
const uploadArea = document.getElementById("uploadArea");
const analyzeBtn = document.getElementById("analyzeBtn");
const removeImageBtn = document.getElementById("removeImageBtn");
const safetyResult = document.getElementById("safetyResult");


// =====================================================
// CHECK ELEMENTS
// =====================================================

console.log("BuildSafe AI started.");

console.log("imageInput:", imageInput);
console.log("imagePreview:", imagePreview);
console.log("analyzeBtn:", analyzeBtn);
console.log("safetyResult:", safetyResult);


// =====================================================
// IMAGE UPLOAD
// =====================================================

if (imageInput) {

    imageInput.addEventListener("change", function(event) {

        const file = event.target.files[0];

        if (!file) {
            return;
        }

        handleImage(file);

    });

}


// =====================================================
// CLICK UPLOAD AREA
// =====================================================

if (uploadArea) {

    uploadArea.addEventListener("click", function(event) {

        // Do not trigger twice when clicking the label
        if (event.target.tagName.toLowerCase() === "label") {
            return;
        }

        imageInput.click();

    });

}


// =====================================================
// HANDLE IMAGE
// =====================================================

function handleImage(file) {

    console.log("Selected file:", file.name);

    // Check image type

    if (!file.type.startsWith("image/")) {

        showSafetyError(
            "Invalid File",
            "Please upload an image file such as JPG, PNG or WEBP."
        );

        return;
    }


    // Create image URL

    const imageURL = URL.createObjectURL(file);

    selectedImage = imageURL;


    // Show preview

    imagePreview.src = imageURL;

    previewContainer.style.display = "block";


    // Hide upload area

    uploadArea.style.display = "none";


    // Enable analyze button

    analyzeBtn.disabled = false;


    // Reset result

    safetyResult.innerHTML = `
        <div class="waiting-icon">🤖</div>

        <h3>Image Ready</h3>

        <p>
            Click "Analyze With AI" to start the AI assessment.
        </p>
    `;


    console.log("Image preview displayed.");

}


// =====================================================
// REMOVE IMAGE
// =====================================================

function removeImage() {

    console.log("Removing image.");

    selectedImage = null;

    imageInput.value = "";

    imagePreview.src = "";

    previewContainer.style.display = "none";

    uploadArea.style.display = "block";

    analyzeBtn.disabled = true;


    safetyResult.innerHTML = `
        <div class="waiting-icon">🔍</div>

        <h3>Waiting for Image</h3>

        <p>
            Upload a construction image and click
            "Analyze With AI".
        </p>
    `;

}


if (removeImageBtn) {

    removeImageBtn.addEventListener(
        "click",
        removeImage
    );

}


// =====================================================
// LOAD AI MODEL
// =====================================================

async function loadSafetyModel() {

    if (safetyClassifier) {

        return safetyClassifier;

    }


    if (modelLoading) {

        return null;

    }


    try {

        modelLoading = true;

        showSafetyLoading(
            "Loading AI model...",
            "The first analysis may take a few minutes."
        );


        console.log("Loading CLIP model...");


        safetyClassifier = await pipeline(
            "zero-shot-image-classification",
            "Xenova/clip-vit-base-patch32"
        );


        console.log("CLIP model loaded successfully.");

        modelLoading = false;

        return safetyClassifier;

    }

    catch (error) {

        console.error(
            "Model loading error:",
            error
        );

        modelLoading = false;

        showSafetyError(
            "AI Model Error",
            "The AI model could not be loaded. Please refresh the page and try again."
        );

        return null;

    }

}


// =====================================================
// ANALYZE SAFETY
// =====================================================

async function analyzeSafety() {

    console.log("Analyze button clicked.");


    // Check image

    if (!selectedImage) {

        showSafetyError(
            "No Image",
            "Please upload a construction-site image first."
        );

        return;

    }


    // Disable button during analysis

    analyzeBtn.disabled = true;

    analyzeBtn.textContent = "⏳ Analyzing...";


    try {

        const classifier = await loadSafetyModel();


        if (!classifier) {

            analyzeBtn.disabled = false;

            analyzeBtn.textContent = "🔍 Analyze With AI";

            return;

        }


        // =================================================
        // STEP 1: CHECK WHETHER IMAGE IS CONSTRUCTION
        // =================================================

        showSafetyLoading(
            "Checking Image...",
            "AI is determining whether this is a construction-site image."
        );


        const constructionLabels = [

            "a construction site with workers, buildings, machinery or construction materials",

            "a non-construction image such as an animal, food, landscape, household object or unrelated scene"

        ];


        console.log(
            "Running construction image validation..."
        );


        const constructionResults = await classifier(
            selectedImage,
            constructionLabels
        );


        console.log(
            "Construction validation:",
            constructionResults
        );


        const constructionResult =
            constructionResults.find(
                result =>
                    result.label.includes(
                        "construction site"
                    )
            );


        const nonConstructionResult =
            constructionResults.find(
                result =>
                    result.label.includes(
                        "non-construction"
                    )
            );


        const constructionScore =
            constructionResult
                ? constructionResult.score
                : 0;


        const nonConstructionScore =
            nonConstructionResult
                ? nonConstructionResult.score
                : 0;


        console.log(
            "Construction score:",
            constructionScore
        );

        console.log(
            "Non-construction score:",
            nonConstructionScore
        );


        // =================================================
        // REJECT NON-CONSTRUCTION IMAGE
        // =================================================

        if (
            nonConstructionScore >
            constructionScore
        ) {

            showSafetyError(
                "Invalid Image",
                "The uploaded image does not appear to show a construction environment."
            );


            safetyResult.innerHTML += `
                <p style="margin-top:15px;">
                    Please upload an image showing a
                    construction site, construction workers,
                    machinery, buildings under construction,
                    materials or related construction activity.
                </p>
            `;


            analyzeBtn.disabled = false;

            analyzeBtn.textContent =
                "🔍 Analyze With AI";


            return;

        }


        // =================================================
        // STEP 2: SAFETY ANALYSIS
        // =================================================

        showSafetyLoading(
            "Analyzing Safety...",
            "AI is assessing visible safety conditions."
        );


        const safetyLabels = [

            "construction workers wearing proper safety helmets and safety vests",

            "construction workers without proper safety equipment",

            "an unsafe construction site",

            "a safe and well protected construction site",

            "a construction site that needs safety attention"

        ];


        console.log(
            "Running safety analysis..."
        );


        const safetyResults = await classifier(
            selectedImage,
            safetyLabels
        );


        console.log(
            "Safety results:",
            safetyResults
        );


        // Get highest score

        const bestResult =
            safetyResults[0];


        if (!bestResult) {

            throw new Error(
                "No AI result was returned."
            );

        }


        displaySafetyResult(
            bestResult
        );


    }

    catch (error) {

        console.error(
            "Analysis error:",
            error
        );


        showSafetyError(
            "Analysis Error",
            "The image could not be analyzed. Please try another image or refresh the page."
        );

    }


    analyzeBtn.disabled = false;

    analyzeBtn.textContent =
        "🔍 Analyze With AI";

}


// =====================================================
// DISPLAY SAFETY RESULT
// =====================================================

function displaySafetyResult(result) {

    const label = result.label;

    const confidence =
        Math.round(
            result.score * 100
        );


    let category = "NEEDS ATTENTION";

    let cssClass = "attention";

    let icon = "⚠️";

    let message =
        "Some safety attention may be required.";


    // HIGH RISK

    if (

        label.includes(
            "without proper safety equipment"
        )

        ||

        label.includes(
            "unsafe construction site"
        )

    ) {

        category = "HIGH RISK";

        cssClass = "high-risk";

        icon = "🚨";

        message =
            "The AI detected visual indicators associated with unsafe construction conditions.";

    }


    // SAFE

    else if (

        label.includes(
            "wearing proper safety helmets"
        )

        ||

        label.includes(
            "safe and well protected"
        )

    ) {

        category = "SAFE";

        cssClass = "safe";

        icon = "✅";

        message =
            "The AI detected visual indicators associated with safer construction conditions.";

    }


    // SHOW RESULT

    safetyResult.innerHTML = `

        <div class="result-box ${cssClass}">

            <div class="result-icon">
                ${icon}
            </div>

            <div class="result-title">
                ${category}
            </div>

            <p>
                ${message}
            </p>

            <div class="confidence">

                <strong>
                    AI Confidence: ${confidence}%
                </strong>

                <div class="confidence-bar">

                    <div
                        class="confidence-fill"
                        style="width:${confidence}%"
                    ></div>

                </div>

            </div>

        </div>

        <p style="margin-top:20px;font-size:13px;">

            AI interpretation:
            <strong>${label}</strong>

        </p>

        <p style="margin-top:12px;font-size:12px;">

            Note: This educational prototype provides
            visual AI assessment and does not replace
            professional safety inspection.

        </p>

    `;

}


// =====================================================
// SAFETY LOADING MESSAGE
// =====================================================

function showSafetyLoading(
    title,
    message
) {

    safetyResult.innerHTML = `

        <div class="waiting-icon">
            ⏳
        </div>

        <h3>
            ${title}
        </h3>

        <p>
            ${message}
        </p>

    `;

}


// =====================================================
// SAFETY ERROR
// =====================================================

function showSafetyError(
    title,
    message
) {

    safetyResult.innerHTML = `

        <div class="result-box high-risk">

            <div class="result-icon">
                ❌
            </div>

            <div class="result-title">
                ${title}
            </div>

            <p>
                ${message}
            </p>

        </div>

    `;

}


// =====================================================
// PROJECT DELAY PREDICTION
// =====================================================

const projectForm =
    document.getElementById(
        "projectForm"
    );


if (projectForm) {

    projectForm.addEventListener(
        "submit",
        function(event) {

            event.preventDefault();

            predictProjectRisk();

        }
    );

}


// =====================================================
// PROJECT RISK FUNCTION
// =====================================================

function predictProjectRisk() {

    const duration =
        Number(
            document.getElementById(
                "duration"
            ).value
        );


    const workers =
        Number(
            document.getElementById(
                "workers"
            ).value
        );


    const budget =
        Number(
            document.getElementById(
                "budget"
            ).value
        );


    const weather =
        Number(
            document.getElementById(
                "weather"
            ).value
        );


    const material =
        Number(
            document.getElementById(
                "material"
            ).value
        );


    const complexity =
        Number(
            document.getElementById(
                "complexity"
            ).value
        );


    // =================================================
    // PROTOTYPE RISK SCORE
    // =================================================

    let riskScore = 0;


    // Long duration

    if (duration > 180) {

        riskScore += 20;

    }

    else if (duration > 120) {

        riskScore += 12;

    }

    else {

        riskScore += 5;

    }


    // Workers

    if (workers < 15) {

        riskScore += 15;

    }

    else if (workers < 25) {

        riskScore += 8;

    }

    else {

        riskScore += 3;

    }


    // Budget

    if (budget < 30) {

        riskScore += 15;

    }

    else if (budget < 50) {

        riskScore += 8;

    }

    else {

        riskScore += 3;

    }


    // Weather

    riskScore +=
        weather * 8;


    // Material

    if (material === 1) {

        riskScore += 18;

    }

    else if (material === 2) {

        riskScore += 9;

    }

    else {

        riskScore += 2;

    }


    // Complexity

    riskScore +=
        complexity * 7;


    // Limit score

    riskScore =
        Math.min(
            riskScore,
            100
        );


    let riskLabel;

    let cssClass;

    let icon;

    let estimatedDuration;


    // HIGH

    if (riskScore >= 60) {

        riskLabel =
            "HIGH DELAY RISK";

        cssClass =
            "high-risk";

        icon = "🚨";

        estimatedDuration =
            Math.round(
                duration * 1.15
            );

    }


    // MEDIUM

    else if (riskScore >= 35) {

        riskLabel =
            "MEDIUM DELAY RISK";

        cssClass =
            "attention";

        icon = "⚠️";

        estimatedDuration =
            Math.round(
                duration * 1.08
            );

    }


    // LOW

    else {

        riskLabel =
            "LOW DELAY RISK";

        cssClass =
            "safe";

        icon = "✅";

        estimatedDuration =
            Math.round(
                duration * 1.03
            );

    }


    const projectResult =
        document.getElementById(
            "projectResult"
        );


    projectResult.innerHTML = `

        <div class="result-box ${cssClass}">

            <div class="result-icon">
                ${icon}
            </div>

            <div class="risk-label">
                ${riskLabel}
            </div>

            <div class="risk-score">
                ${riskScore}%
            </div>

            <p>
                Estimated project duration:
                <strong>
                    ${estimatedDuration} days
                </strong>
            </p>

        </div>

        <p style="margin-top:20px;font-size:13px;">

            The prediction is based on project duration,
            workforce, budget, weather risk, material
            availability and project complexity.

        </p>

        <p style="margin-top:12px;font-size:12px;">

            Note: This is a rule-based educational
            prototype and is not a validated commercial
            prediction model.

        </p>

    `;

}


// =====================================================
// DRAG AND DROP
// =====================================================

if (uploadArea) {

    uploadArea.addEventListener(
        "dragover",
        function(event) {

            event.preventDefault();

            uploadArea.style.borderColor =
                "#f59e0b";

        }
    );


    uploadArea.addEventListener(
        "dragleave",
        function() {

            uploadArea.style.borderColor =
                "#cbd5e1";

        }
    );


    uploadArea.addEventListener(
        "drop",
        function(event) {

            event.preventDefault();


            uploadArea.style.borderColor =
                "#cbd5e1";


            const file =
                event.dataTransfer.files[0];


            if (file) {

                handleImage(file);

            }

        }
    );

}


// =====================================================
// CONNECT BUTTON TO FUNCTION
// =====================================================

if (analyzeBtn) {

    analyzeBtn.addEventListener(
        "click",
        analyzeSafety
    );

}


// =====================================================
// MAKE FUNCTIONS AVAILABLE
// =====================================================

window.analyzeSafety =
    analyzeSafety;

window.removeImage =
    removeImage;


// =====================================================
// FINAL MESSAGE
// =====================================================

console.log(
    "BuildSafe AI loaded successfully."
);