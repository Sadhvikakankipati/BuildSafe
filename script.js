import { pipeline, env } from
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

/*
=========================================================
BuildSafe AI
GitHub Pages Version
=========================================================

Safety:
CLIP zero-shot image classification

Project Planning:
Browser-based risk scoring prototype

No Flask
No Python
No Render
=========================================================
*/


// ======================================================
// HUGGING FACE CONFIGURATION
// ======================================================

env.allowLocalModels = false;
env.useBrowserCache = true;


// ======================================================
// GLOBAL VARIABLES
// ======================================================

let safetyClassifier = null;
let modelLoading = false;


// ======================================================
// IMAGE PREVIEW
// ======================================================

const imageInput =
    document.getElementById("safetyImage");

const imagePreview =
    document.getElementById("imagePreview");

const imagePreviewContainer =
    document.getElementById(
        "imagePreviewContainer"
    );

const dropArea =
    document.getElementById("dropArea");

const analyzeButton =
    document.getElementById(
        "analyzeButton"
    );


// When user selects image
imageInput.addEventListener(
    "change",
    function () {

        const file = this.files[0];

        if (!file) {
            return;
        }

        if (!file.type.startsWith("image/")) {

            alert(
                "Please select a valid image."
            );

            return;
        }

        const reader =
            new FileReader();

        reader.onload =
            function (event) {

                imagePreview.src =
                    event.target.result;

                imagePreviewContainer
                    .classList
                    .remove("hidden");

                dropArea
                    .classList
                    .add("hidden");

            };

        reader.readAsDataURL(file);
    }
);


// ======================================================
// REMOVE IMAGE
// ======================================================

function removeImage() {

    imageInput.value = "";

    imagePreview.src = "";

    imagePreviewContainer
        .classList
        .add("hidden");

    dropArea
        .classList
        .remove("hidden");

    document.getElementById(
        "safetyResult"
    ).innerHTML = `

        <div class="empty-result">

            <div class="empty-icon">
                🔍
            </div>

            <h3>Waiting for Image</h3>

            <p>
                Upload a construction image and
                click "Analyze With AI".
            </p>

        </div>

    `;
}


// ======================================================
// LOAD CLIP MODEL
// ======================================================

async function loadSafetyModel() {

    if (safetyClassifier) {
        return safetyClassifier;
    }

    if (modelLoading) {

        while (modelLoading) {

            await new Promise(
                resolve =>
                    setTimeout(
                        resolve,
                        300
                    )
            );

        }

        return safetyClassifier;
    }

    modelLoading = true;

    const status =
        document.getElementById(
            "modelStatus"
        );

    status.textContent =
        "Loading AI model... This may take some time the first time.";

    analyzeButton.disabled = true;

    try {

        safetyClassifier =
            await pipeline(
                "zero-shot-image-classification",
                "Xenova/clip-vit-base-patch32"
            );

        status.textContent =
            "AI model ready.";

        return safetyClassifier;

    }

    catch (error) {

        console.error(error);

        status.textContent =
            "Unable to load AI model.";

        throw error;

    }

    finally {

        modelLoading = false;

        analyzeButton.disabled = false;

    }
}


// ======================================================
// SAFETY ANALYSIS
// ======================================================

async function analyzeSafety() {

    const file = imageInput.files[0];

    if (!file) {
        alert("Please upload a construction-site image first.");
        return;
    }

    const resultContainer =
        document.getElementById("safetyResult");

    resultContainer.innerHTML = `
        <div class="empty-result">
            <div class="empty-icon">🤖</div>
            <h3>Checking Image...</h3>
            <p>
                AI is first checking whether this is a
                construction-site image.
            </p>
        </div>
    `;

    try {

        const classifier = await loadSafetyModel();

        /*
        =====================================================
        STEP 1 — CHECK WHETHER IMAGE IS CONSTRUCTION RELATED
        =====================================================
        */

        const constructionLabels = [

            "a construction site",
            "a construction building site",
            "construction workers at a building site",
            "construction equipment at a construction site",
            "an image unrelated to construction"

        ];

        const constructionResults =
            await classifier(
                imagePreview.src,
                constructionLabels
            );

        const bestConstruction =
            constructionResults[0];

        /*
        =====================================================
        STEP 2 — REJECT NON-CONSTRUCTION IMAGES
        =====================================================
        */

        const constructionScore =
            constructionResults
                .filter(item =>
                    item.label !==
                    "an image unrelated to construction"
                )
                .reduce(
                    (sum, item) =>
                        sum + item.score,
                    0
                );

        const unrelatedResult =
            constructionResults.find(
                item =>
                    item.label ===
                    "an image unrelated to construction"
            );

        const unrelatedScore =
            unrelatedResult
                ? unrelatedResult.score
                : 0;


        /*
        A relatively strict threshold is used so that
        obvious non-construction images are rejected.
        */

        if (
            unrelatedScore > constructionScore ||
            (
                bestConstruction.score < 0.35 &&
                unrelatedScore > 0.20
            )
        ) {

            resultContainer.innerHTML = `

                <div class="empty-result">

                    <div class="empty-icon">
                        🚫
                    </div>

                    <h3>
                        Invalid Image
                    </h3>

                    <p>
                        Please upload a construction-site
                        image.
                    </p>

                    <div class="recommendation">

                        <strong>
                            ⚠️ Image Not Supported
                        </strong>

                        <p>
                            The uploaded image does not appear
                            to show a construction environment.
                            Safety analysis is available only
                            for construction-site images.
                        </p>

                    </div>

                </div>

            `;

            return;
        }


        /*
        =====================================================
        STEP 3 — CONSTRUCTION IMAGE CONFIRMED
        =====================================================
        */

        resultContainer.innerHTML = `

            <div class="empty-result">

                <div class="empty-icon">
                    🏗️
                </div>

                <h3>
                    Construction Image Detected
                </h3>

                <p>
                    Running construction safety analysis...
                </p>

            </div>

        `;


        /*
        =====================================================
        STEP 4 — SAFETY CLASSIFICATION
        =====================================================
        */

        const safetyLabels = [

            "construction workers wearing safety helmets and safety vests",

            "construction workers without proper safety equipment",

            "an unsafe construction site",

            "a safe and well protected construction site"

        ];


        const safetyResults =
            await classifier(
                imagePreview.src,
                safetyLabels
            );


        displaySafetyResult(
            safetyResults
        );

    }

    catch (error) {

        console.error(
            "Safety analysis error:",
            error
        );

        resultContainer.innerHTML = `

            <div class="empty-result">

                <div class="empty-icon">
                    ⚠️
                </div>

                <h3>
                    Analysis Error
                </h3>

                <p>
                    The image could not be analyzed.
                    Please try another construction-site
                    image.
                </p>

            </div>

        `;

    }

}


// ======================================================
// DISPLAY SAFETY RESULT
// ======================================================

function displaySafetyResult(results) {

    const container =
        document.getElementById(
            "safetyResult"
        );


    if (!results || results.length === 0) {

        container.innerHTML = `

            <div class="empty-result">
                <h3>No Result</h3>
            </div>

        `;

        return;
    }


    const best =
        results[0];


    const label =
        best.label.toLowerCase();


    let status =
        "NEEDS ATTENTION";

    let statusClass =
        "medium-risk";

    let recommendation =
        "A further safety inspection is recommended.";


    if (
        label.includes("without") ||
        label.includes("unsafe")
    ) {

        status =
            "HIGH RISK";

        statusClass =
            "high-risk";

        recommendation =
            "Immediate safety inspection recommended. Review PPE, fall protection and site safety controls.";

    }

    else if (
        label.includes("wearing") ||
        label.includes("safe")
    ) {

        status =
            "SAFE";

        statusClass =
            "safe-risk";

        recommendation =
            "Visible safety conditions appear acceptable, but routine inspection should continue.";

    }


    const confidence =
        best.score * 100;


    let bars = "";


    results.forEach(
        function (item) {

            const percentage =
                item.score * 100;

            bars += `

                <div class="label-item">

                    <div class="label-name">

                        <span>
                            ${item.label}
                        </span>

                        <strong>
                            ${percentage.toFixed(1)}%
                        </strong>

                    </div>

                    <div class="mini-bar">

                        <div
                            class="mini-fill"
                            style="width:${percentage}%">
                        </div>

                    </div>

                </div>

            `;

        }
    );


    container.innerHTML = `

        <div class="risk-result">

            <div class="risk-badge ${statusClass}">
                ${status}
            </div>

            <h3>
                AI Safety Assessment
            </h3>

            <p>
                Most likely interpretation:
            </p>

            <strong>
                ${best.label}
            </strong>


            <div class="confidence">

                <div>
                    AI Confidence:
                    <strong>
                        ${confidence.toFixed(2)}%
                    </strong>
                </div>

                <div class="confidence-bar">

                    <div
                        class="confidence-fill"
                        style="width:${confidence}%">
                    </div>

                </div>

            </div>


            <div class="recommendation">

                <strong>
                    Recommendation
                </strong>

                <p>
                    ${recommendation}
                </p>

            </div>


            <div class="label-list">

                <strong>
                    AI Classification Scores
                </strong>

                <br><br>

                ${bars}

            </div>

        </div>

    `;

}


// ======================================================
// PROJECT PLANNING
// ======================================================

function predictProject() {

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


    // ==================================================
    // VALIDATION
    // ==================================================

    if (
        duration <= 0 ||
        workers <= 0 ||
        budget <= 0
    ) {

        alert(
            "Please enter valid project values."
        );

        return;
    }


    if (
        weather < 1 ||
        weather > 10 ||
        material < 1 ||
        material > 10 ||
        complexity < 1 ||
        complexity > 10
    ) {

        alert(
            "Weather, material availability and complexity must be between 1 and 10."
        );

        return;
    }


    /*
    =====================================================
    PROJECT RISK SCORING

    This is a browser-based prototype.

    Higher values:
    - Weather risk
    - Complexity

    Lower material availability:
    - increases risk

    Very long projects:
    - increase risk

    Very small workforce for a long project:
    - increases risk
    =====================================================
    */


    let riskScore = 0;


    // Duration contribution

    if (duration > 180) {

        riskScore += 10;

    }

    else if (duration > 120) {

        riskScore += 7;

    }

    else if (duration > 60) {

        riskScore += 4;

    }


    // Worker contribution

    if (
        workers < 10 &&
        duration > 120
    ) {

        riskScore += 6;

    }

    else if (workers < 15) {

        riskScore += 3;

    }


    // Budget contribution

    if (budget < 250000) {

        riskScore += 5;

    }

    else if (budget < 500000) {

        riskScore += 3;

    }


    // Weather contribution

    riskScore +=
        weather * 1.5;


    // Material availability

    riskScore +=
        (10 - material) * 1.5;


    // Complexity

    riskScore +=
        complexity * 1.5;


    // Normalize approximately

    let probability =
        Math.min(
            95,
            Math.max(
                5,
                20 + riskScore
            )
        );


    let highRisk =
        probability >= 50;


    let riskText =
        highRisk
            ? "HIGH DELAY RISK"
            : "LOW DELAY RISK";


    let riskClass =
        highRisk
            ? "high"
            : "low";


    // Estimated duration

    let estimatedDuration;


    if (highRisk) {

        estimatedDuration =
            Math.round(
                duration * 1.15
            );

    }

    else {

        estimatedDuration =
            Math.round(
                duration * 1.03
            );

    }


    // Recommendation

    let recommendation;


    if (highRisk) {

        recommendation =
            "Consider additional resources, better material planning, weather contingency and closer project monitoring.";

    }

    else {

        recommendation =
            "Project conditions appear relatively manageable. Continue monitoring resources, materials and weather.";

    }


    displayProjectResult(
        riskText,
        riskClass,
        probability,
        estimatedDuration,
        recommendation
    );

}


// ======================================================
// DISPLAY PROJECT RESULT
// ======================================================

function displayProjectResult(
    riskText,
    riskClass,
    probability,
    estimatedDuration,
    recommendation
) {

    const container =
        document.getElementById(
            "projectResult"
        );


    container.innerHTML = `

        <div class="project-result">

            <div class="project-risk ${riskClass}">
                ${riskText}
            </div>

            <div class="project-probability">

                Estimated risk probability:
                <strong>
                    ${probability.toFixed(1)}%
                </strong>

            </div>


            <div class="project-stats">

                <div class="stat">

                    <strong>
                        ${estimatedDuration}
                    </strong>

                    <span>
                        Estimated Days
                    </span>

                </div>


                <div class="stat">

                    <strong>
                        ${probability.toFixed(0)}%
                    </strong>

                    <span>
                        Delay Risk
                    </span>

                </div>

            </div>


            <div class="recommendation">

                <strong>
                    AI Recommendation
                </strong>

                <p>
                    ${recommendation}
                </p>

            </div>

        </div>

    `;

}


// ======================================================
// DRAG AND DROP
// ======================================================

const dropZone =
    document.getElementById(
        "dropArea"
    );


dropZone.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        dropZone.style.borderColor =
            "#5d8f35";

    }
);


dropZone.addEventListener(
    "dragleave",
    function () {

        dropZone.style.borderColor =
            "#cbd7c5";

    }
);


dropZone.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        dropZone.style.borderColor =
            "#cbd7c5";


        const files =
            event.dataTransfer.files;


        if (
            files.length > 0 &&
            files[0].type.startsWith("image/")
        ) {

            imageInput.files =
                files;

            imageInput.dispatchEvent(
                new Event("change")
            );

        }

    }
);


// ======================================================
// INITIAL MESSAGE
// ======================================================

console.log(
    "BuildSafe AI initialized successfully."
);