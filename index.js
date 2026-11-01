
// ======================================
// 1. Select HTML Elements
// ======================================

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const clearBtn = document.getElementById("clearBtn");
const recognizeBtn = document.getElementById("recognizeBtn");
const trainBtn = document.getElementById("trainBtn");

const statusText = document.getElementById("status");

const predictionText = document.getElementById("prediction");
const messageText = document.getElementById("message");

const missingText = document.getElementById("missing");
const mistakenText = document.getElementById("mistaken");

const confidenceText = document.getElementById("confidence");
const progressBar = document.getElementById("progressBar");


// ======================================
// 2. Canvas Settings
// ======================================

ctx.fillStyle = "white";
ctx.fillRect(0, 0, 280, 280);

ctx.strokeStyle = "#111827";
ctx.lineWidth = 14;
ctx.lineCap = "round";
ctx.lineJoin = "round";

let drawing = false;
let hasDrawing = false;


// ======================================
// 3. Drawing Functions
// ======================================

function getPosition(event) {

    const rect = canvas.getBoundingClientRect();

    return {
        x: (event.clientX - rect.left) * canvas.width / rect.width,
        y: (event.clientY - rect.top) * canvas.height / rect.height
    };

}

canvas.addEventListener("pointerdown", (event) => {

    drawing = true;
    hasDrawing = true;

    const pos = getPosition(event);

    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);

    canvas.setPointerCapture(event.pointerId);

});

canvas.addEventListener("pointermove", (event) => {

    if (!drawing) return;

    const pos = getPosition(event);

    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

});

function stopDrawing() {
    drawing = false;
}

canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);


// ======================================
// 4. Clear Canvas
// ======================================

clearBtn.addEventListener("click", () => {

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, 280, 280);

    hasDrawing = false;

    predictionText.textContent = "?";
    messageText.textContent = "Draw a symbol to recognize it.";

    missingText.textContent = "Not Checked";
    mistakenText.textContent = "Not Checked";

    confidenceText.textContent = "0%";
    progressBar.style.width = "0%";

});


// ======================================
// 5. Generate Training Data
// ======================================

// Labels:
// 0 = X
// 1 = O
// 2 = Invalid

function generateSample(label) {

    const c = document.createElement("canvas");

    c.width = 28;
    c.height = 28;

    const g = c.getContext("2d");

    g.fillStyle = "white";
    g.fillRect(0, 0, 28, 28);

    g.save();

    g.translate(
        14 + (Math.random() - 0.5) * 3,
        14 + (Math.random() - 0.5) * 3
    );

    g.rotate((Math.random() - 0.5) * 0.35);

    g.strokeStyle = "black";
    g.lineWidth = 2.5;
    g.lineCap = "round";
    g.lineJoin = "round";

    g.beginPath();

    if (label === 0) {

        // X

        g.moveTo(-8, -8);
        g.lineTo(8, 8);

        g.moveTo(8, -8);
        g.lineTo(-8, 8);

    } else if (label === 1) {

        // O

        g.ellipse(
            0, 0,
            7 + Math.random() * 1.5,
            9 + Math.random() * 1.5,
            0, 0, Math.PI * 2
        );

    } else {

        // Invalid shapes

        const shapes = [
            [[-8, -8], [-8, 8]],
            [[-8, -8], [8, -8]],
            [[-8, 0], [8, 0]],
            [[-8, 8], [8, -8]],
            [[-8, -8], [0, 0], [8, -8]],
            [[-8, 8], [0, -8], [8, 8]]
        ];

        const shape = shapes[
            Math.floor(Math.random() * shapes.length)
        ];

        g.moveTo(shape[0][0], shape[0][1]);

        for (let i = 1; i < shape.length; i++) {
            g.lineTo(shape[i][0], shape[i][1]);
        }

    }

    g.stroke();
    g.restore();

    const image = g.getImageData(0, 0, 28, 28);

    const values = [];

    for (let i = 0; i < image.data.length; i += 4) {

        values.push((255 - image.data[i]) / 255);

    }

    return values;

}


// ======================================
// 6. Create Neural Network
// ======================================

let model = null;

function createModel() {

    model = tf.sequential();

    model.add(tf.layers.dense({
        inputShape: [784],
        units: 64,
        activation: "relu"
    }));

    model.add(tf.layers.dense({
        units: 32,
        activation: "relu"
    }));

    model.add(tf.layers.dense({
        units: 3,
        activation: "softmax"
    }));

    model.compile({

        optimizer: tf.train.adam(0.001),

        loss: "categoricalCrossentropy",

        metrics: ["accuracy"]

    });

}


// ======================================
// 7. Train Neural Network
// ======================================

async function trainNetwork() {

    trainBtn.disabled = true;
    recognizeBtn.disabled = true;

    statusText.textContent = "Training Neural Network...";

    let xs;
    let ys;

    try {

        if (model) {
            model.dispose();
        }

        createModel();

        const inputs = [];
        const labels = [];

        for (let i = 0; i < 500; i++) {

            // X
            inputs.push(generateSample(0));
            labels.push([1, 0, 0]);

            // O
            inputs.push(generateSample(1));
            labels.push([0, 1, 0]);

            // Invalid
            inputs.push(generateSample(2));
            labels.push([0, 0, 1]);

        }

        xs = tf.tensor2d(inputs);
        ys = tf.tensor2d(labels);

        await model.fit(xs, ys, {

            epochs: 20,

            batchSize: 32,

            shuffle: true,

            callbacks: {

                onEpochEnd: async (epoch) => {

                    statusText.textContent =
                        `Training: Epoch ${epoch + 1}/20`;

                    await tf.nextFrame();

                }

            }

        });

        statusText.textContent = "Network Trained Successfully!";

        document.getElementById("networkStatus").textContent = "Ready";

        recognizeBtn.disabled = false;

    } catch (error) {

        statusText.textContent = "Training Failed!";

        console.error(error);

    } finally {

        if (xs) xs.dispose();
        if (ys) ys.dispose();

        trainBtn.disabled = false;

    }

}


// ======================================
// 8. Convert Drawing to 28x28 Data
// ======================================

function getDrawingData() {

    const smallCanvas = document.createElement("canvas");

    smallCanvas.width = 28;
    smallCanvas.height = 28;

    const smallCtx = smallCanvas.getContext("2d");

    smallCtx.drawImage(canvas, 0, 0, 28, 28);

    const image = smallCtx.getImageData(0, 0, 28, 28);

    const values = [];

    let pixelCount = 0;

    for (let i = 0; i < image.data.length; i += 4) {

        const value = (255 - image.data[i]) / 255;

        values.push(value);

        if (value > 0.15) {
            pixelCount++;
        }

    }

    return {
        values,
        pixelCount
    };

}


// ======================================
// 9. Recognize X, O or Invalid
// ======================================

recognizeBtn.addEventListener("click", async () => {

    if (!model || !hasDrawing) return;

    const data = getDrawingData();

    // Missing Data

    if (data.pixelCount < 8) {

        predictionText.textContent = "?";
        messageText.textContent = "Missing Data Detected!";

        missingText.textContent = "Detected";
        mistakenText.textContent = "Not Checked";

        confidenceText.textContent = "0%";
        progressBar.style.width = "0%";

        return;

    }

    missingText.textContent = "Not Detected";

    const input = tf.tensor2d([data.values]);

    const output = model.predict(input);

    const probabilities = await output.data();

    const xConfidence = probabilities[0];
    const oConfidence = probabilities[1];
    const invalidConfidence = probabilities[2];

    input.dispose();
    output.dispose();

    const maxConfidence = Math.max(
        xConfidence,
        oConfidence,
        invalidConfidence
    );

    const percentage = Math.round(maxConfidence * 100);

    confidenceText.textContent = percentage + "%";
    progressBar.style.width = percentage + "%";

    // Invalid Data

    if (
        invalidConfidence >= xConfidence &&
        invalidConfidence >= oConfidence
    ) {

        predictionText.textContent = "?";
        messageText.textContent = "Invalid Data Detected!";

        mistakenText.textContent = "Detected";

        return;

    }

    // X Detection

    if (xConfidence > oConfidence) {

        predictionText.textContent = "X";
        messageText.textContent = "X Recognized Successfully!";

    }

    // O Detection

    else {

        predictionText.textContent = "O";
        messageText.textContent = "O Recognized Successfully!";

    }

    mistakenText.textContent = "Not Detected";

});


// ======================================
// 10. Start Training
// ======================================

trainBtn.addEventListener("click", trainNetwork);


// ======================================
// 11. Initialize Application
// ======================================

async function initialize() {

    statusText.textContent = "Preparing Neural Network...";

    try {

        await tf.ready();

        await trainNetwork();

    } catch (error) {

        statusText.textContent = "Error: " + error.message;

        console.error(error);

    }

}

initialize();
