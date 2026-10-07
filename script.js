import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;
env.useBrowserCache = true;

let safetyClassifier = null;
let selectedImage = null;
let modelLoading = false;

// ===============================
// IMAGE UPLOAD
// ===============================

const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const previewContainer = document.getElementById("previewContainer");
const uploadArea = document.getElementById("uploadArea");
const analyzeBtn = document.getElementById("analyzeBtn");
const removeImageBtn = document.getElementById("removeImageBtn");

if (imageInput) {
    imageInput.addEventListener("change", handleImageUpload);
}

if (removeImageBtn) {
    removeImageBtn.addEventListener("click", removeImage);
}

if (analyzeBtn) {
    analyzeBtn.addEventListener("click", analyzeSafety);
}

function handleImageUpload(event) {
    const file = event.target.files[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
        showSafetyError(
            "Invalid File",
            "Please upload an image file such as JPG, JPEG, PNG or WEBP."
        );
        return;
    }

    selectedImage = file;

    const reader = new FileReader();

    reader.onload = function (e) {
        imagePreview.src = e.target.result;

        if (previewContainer) {
            previewContainer.style.display = "block";
        }

        if (uploadArea) {
            uploadArea.style.display = "none";
        }

        if (analyzeBtn) {
            analyzeBtn.disabled = false;
        }

        clearSafetyResult();
    };

    reader.readAsDataURL(file);
}

function removeImage() {
    selectedImage = null;

    if (imageInput) {
        imageInput.value = "";
    }

    if (imagePreview) {
        imagePreview.src = "";
    }

    if (previewContainer) {
        previewContainer.style.display = "none";
    }

    if (uploadArea) {
        uploadArea.style.display = "block";
    }

    if (analyzeBtn) {
        analyzeBtn.disabled = true;
    }

    clearSafetyResult();
}


// ===============================
// LOAD CLIP MODEL
// ===============================

async function loadSafetyModel() {

    if (safetyClassifier) {
        return safetyClassifier;
    }

    if (modelLoading) {
        return null;
    }

    modelLoading = true;

    try {

        showSafetyLoading(
            "Loading AI model...",
            "The first analysis may take a little longer while the CLIP model downloads."
        );

        safetyClassifier = await pipeline(
            "zero-shot-image-classification",
            "Xenova/clip-vit-base-patch32"
        );

        modelLoading = false;

        return safetyClassifier;

    } catch (error) {

        modelLoading = false;

        console.error("Model loading error:", error);

        showSafetyError(
            "AI Model Error",
            "The AI model could not be loaded. Please check your internet connection and refresh the page."
        );

        return null;
    }
}


// ===============================
// SAFETY ANALYSIS
// ===============================

async function analyzeSafety() {

    if (!selectedImage) {

        showSafetyError(
            "No Image Selected",
            "Please upload a construction-site image first."
        );

        return;
    }

    try {

        if (analyzeBtn) {
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = "⏳ Analyzing...";
        }

        showSafetyLoading(
            "Analyzing image...",
            "AI is checking whether this is a construction image and assessing visible safety conditions."
        );

        const classifier = await loadSafetyModel();

        if (!classifier) {
            return;
        }

        // ---------------------------------
        // STEP 1: CHECK CONSTRUCTION IMAGE
        // ---------------------------------

        const constructionLabels = [
            "a construction site with workers, buildings, machinery or construction materials",
            "a non-construction image such as an animal, food, person, landscape or household object"
        ];

        const constructionResults = await classifier(
            imagePreview.src,
            constructionLabels
        );

        console.log("Construction validation:", constructionResults);

        const topConstructionResult = constructionResults[0];

        const isConstruction =
            topConstructionResult.label.includes("construction site");

        // Reject non-construction images
        if (!isConstruction) {

            showSafetyError(
                "Invalid Image",
                "Please upload a construction-site image."
            );

            showSafetyMessage(
                "The uploaded image does not appear to show a construction environment. Safety analysis is available only for construction-site images."
            );

            return;
        }

        // ---------------------------------
        // STEP 2: SAFETY ANALYSIS
        // ---------------------------------

        const safetyLabels = [

            "construction workers wearing proper safety helmets and safety vests",

            "construction workers without proper safety equipment",

            "an unsafe construction site",

            "a safe and well protected construction site",

            "a construction site that needs safety attention"

        ];

        showSafetyLoading(
            "Construction site detected ✓",
            "Now analyzing the visible safety conditions..."
        );

        const safetyResults = await classifier(
            imagePreview.src,
            safetyLabels
        );

        console.log("Safety results:", safetyResults);

        displaySafetyResult(safetyResults);

    } catch (error) {

        console.error("Safety analysis error:", error);

        showSafetyError(
            "Analysis Error",
            "Something went wrong while analyzing the image. Please refresh the page and try again."
        );

    } finally {

        if (analyzeBtn) {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = "🔍 Analyze Safety";
        }
    }
}


// ===============================
// DISPLAY SAFETY RESULT
// ===============================

function displaySafetyResult(results) {

    if (!results || results.length === 0) {

        showSafetyError(
            "No Result",
            "The AI could not classify this image."
        );

        return;
    }

    const best = results[0];

    let status = "NEEDS ATTENTION";
    let icon = "⚠️";
    let message =
        "Some visible safety conditions may require attention.";

    const label = best.label.toLowerCase();

    if (
        label.includes("without proper") ||
        label.includes("unsafe")
    ) {

        status = "HIGH RISK";
        icon = "🚨";

        message =
            "The AI detected visible indicators associated with an unsafe construction environment.";

    } else if (
        label.includes("wearing proper") ||
        label.includes("safe and well")
    ) {

        status = "SAFE";
        icon = "✅";

        message =
            "The AI detected visible indicators of safer construction practices.";

    }

    const confidence = Math.round(best.score * 100);

    let resultHTML = `
        <div class="result-card">

            <div class="result-icon">
                ${icon}
            </div>

            <h3>${status}</h3>

            <p>${message}</p>

            <div class="confidence">
                <strong>AI Confidence: ${confidence}%</strong>

                <div class="progress-bar">
                    <div
                        class="progress-fill"
                        style="width:${confidence}%"
                    ></div>
                </div>
            </div>

            <h4>AI Classification</h4>
    `;

    results.forEach(result => {

        const score = Math.round(result.score * 100);

        resultHTML += `
            <div class="classification-row">

                <div class="classification-label">
                    <span>${result.label}</span>
                    <strong>${score}%</strong>
                </div>

                <div class="progress-bar">
                    <div
                        class="progress-fill"
                        style="width:${score}%"
                    ></div>
                </div>

            </div>
        `;
    });

    resultHTML += `
            <div class="ai-note">
                <strong>Note:</strong>
                This is an AI-based visual assessment prototype.
                It should not replace professional construction-site safety inspection.
            </div>

        </div>
    `;

    const resultContainer =
        document.getElementById("safetyResult");

    if (resultContainer) {
        resultContainer.innerHTML = resultHTML;
        resultContainer.style.display = "block";
    }
}


// ===============================
// SAFETY UI HELPERS
// ===============================

function showSafetyLoading(title, message) {

    const resultContainer =
        document.getElementById("safetyResult");

    if (!resultContainer) return;

    resultContainer.style.display = "block";

    resultContainer.innerHTML = `
        <div class="result-card">

            <div class="result-icon">🤖</div>

            <h3>${title}</h3>

            <p>${message}</p>

            <div class="loading-spinner"></div>

        </div>
    `;
}

function showSafetyError(title, message) {

    const resultContainer =
        document.getElementById("safetyResult");

    if (!resultContainer) return;

    resultContainer.style.display = "block";

    resultContainer.innerHTML = `
        <div class="result-card error-result">

            <div class="result-icon">🚫</div>

            <h3>${title}</h3>

            <p>${message}</p>

        </div>
    `;
}

function showSafetyMessage(message) {

    const resultContainer =
        document.getElementById("safetyResult");

    if (!resultContainer) return;

    resultContainer.innerHTML += `
        <p class="safety-message">
            ${message}
        </p>
    `;
}

function clearSafetyResult() {

    const resultContainer =
        document.getElementById("safetyResult");

    if (resultContainer) {
        resultContainer.innerHTML = "";
        resultContainer.style.display = "none";
    }
}


// ===============================
// PROJECT DELAY PREDICTION
// ===============================

const projectForm =
    document.getElementById("projectForm");

if (projectForm) {
    projectForm.addEventListener(
        "submit",
        predictProject
    );
}

function predictProject(event) {

    event.preventDefault();

    const duration =
        Number(document.getElementById("duration").value);

    const workers =
        Number(document.getElementById("workers").value);

    const budget =
        Number(document.getElementById("budget").value);

    const weather =
        Number(document.getElementById("weather").value);

    const material =
        Number(document.getElementById("material").value);

    const complexity =
        Number(document.getElementById("complexity").value);

    let riskScore = 0;

    // Duration
    if (duration > 180) {
        riskScore += 10;
    } else if (duration > 120) {
        riskScore += 7;
    } else if (duration > 60) {
        riskScore += 4;
    }

    // Workers
    if (workers < 10 && duration > 120) {
        riskScore += 6;
    } else if (workers < 15) {
        riskScore += 3;
    }

    // Budget
    if (budget < 250000) {
        riskScore += 5;
    } else if (budget < 500000) {
        riskScore += 3;
    }

    // Weather
    riskScore += weather * 1.5;

    // Material availability
    riskScore += (10 - material) * 1.5;

    // Complexity
    riskScore += complexity * 1.5;

    let probability =
        Math.min(
            95,
            Math.max(
                5,
                20 + riskScore
            )
        );

    const highRisk =
        probability >= 50;

    const estimatedDuration =
        highRisk
            ? Math.round(duration * 1.15)
            : Math.round(duration * 1.03);

    let status;
    let icon;
    let recommendation;

    if (highRisk) {

        status = "HIGH DELAY RISK";
        icon = "🚨";

        recommendation =
            "Consider increasing resources, improving material availability, and preparing for weather-related disruptions.";

    } else {

        status = "LOW DELAY RISK";
        icon = "✅";

        recommendation =
            "The project currently shows relatively low delay risk. Continue monitoring resources, materials and weather.";
    }

    const resultContainer =
        document.getElementById("projectResult");

    if (!resultContainer) return;

    resultContainer.style.display = "block";

    resultContainer.innerHTML = `

        <div class="result-card">

            <div class="result-icon">
                ${icon}
            </div>

            <h3>${status}</h3>

            <p>
                Estimated probability of project delay:
                <strong>${Math.round(probability)}%</strong>
            </p>

            <p>
                Estimated project duration:
                <strong>${estimatedDuration} days</strong>
            </p>

            <div class="confidence">

                <strong>Delay Risk</strong>

                <div class="progress-bar">

                    <div
                        class="progress-fill"
                        style="width:${probability}%"
                    ></div>

                </div>

            </div>

            <div class="ai-note">

                <strong>Recommendation:</strong><br>

                ${recommendation}

            </div>

            <div class="ai-note">

                <strong>Prototype note:</strong>
                This prediction uses a rule-based AI prototype
                for demonstration purposes and is not a
                production project-management model.

            </div>

        </div>
    `;
}


// ===============================
// DRAG AND DROP
// ===============================

if (uploadArea) {

    uploadArea.addEventListener(
        "dragover",
        function (event) {

            event.preventDefault();

            uploadArea.classList.add("drag-over");
        }
    );

    uploadArea.addEventListener(
        "dragleave",
        function () {

            uploadArea.classList.remove("drag-over");
        }
    );

    uploadArea.addEventListener(
        "drop",
        function (event) {

            event.preventDefault();

            uploadArea.classList.remove("drag-over");

            const file =
                event.dataTransfer.files[0];

            if (!file) return;

            if (!file.type.startsWith("image/")) {

                showSafetyError(
                    "Invalid File",
                    "Please upload an image file."
                );

                return;
            }

            selectedImage = file;

            const reader =
                new FileReader();

            reader.onload = function (e) {

                imagePreview.src =
                    e.target.result;

                if (previewContainer) {
                    previewContainer.style.display =
                        "block";
                }

                uploadArea.style.display =
                    "none";

                if (analyzeBtn) {
                    analyzeBtn.disabled = false;
                }

                clearSafetyResult();
            };

            reader.readAsDataURL(file);
        }
    );
}

console.log(
    "BuildSafe AI loaded successfully."
);
window.analyzeSafety = analyzeSafety;
window.removeImage = removeImage;