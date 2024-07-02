async function loadModel(modelPath) {
    try {
        // Load the ONNX model
        const model = await ort.InferenceSession.create(modelPath);
        return model;
    } catch (error) {
        console.error('Error loading model:', error);
        throw error;
    }
}

async function predictImage() {
    const fileInput = document.getElementById('image-upload');
	const fileURLInput = document.getElementById('image-url');
    const predictionResult = document.getElementById('prediction-result');
    const uploadedImage = document.getElementById('uploaded-image');

    // Check if image was uploaded via file input
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const reader = new FileReader();

        reader.onload = async function(event) {
            imageData = event.target.result;
            predictImage(imageData);
        };

        reader.readAsDataURL(file);
    } else {
        predictionResult.innerHTML = 'Please upload an image.';
    }

    async function predictImage(imageData) {
        uploadedImage.src = imageData;
        uploadedImage.style.display = 'block';

        const img = new Image();
        img.crossOrigin = "anonymous";
        img.src = imageData;

        img.onload = async function() {
            try {
                const model_path = 'model/goat.onnx';
                const model = await loadModel(model_path);

                const tensor = await preprocessImage(img);
                const input = { 'input': tensor };
                const output = await model.run(input);
                const predictions = output.dense.cpuData[0];
				const prediction_pct = predictions > 0.5 ? predictions : 1.0 - predictions;
                const predictionText = predictions > 0.5 ? "Upgoat" : "Downgoat";
				const objside = predictionText === "Upgoat" ? "right-side-up" : "up-side-down";
                predictionResult.innerHTML = `If it is a goat then it must be a <b>${predictionText}</b>. (I am <b>${(prediction_pct * 100).toFixed(0)}%</b> sure!) At least it is ${objside}.`;

            } catch (error) {
                console.error('Error making prediction:', error);
                predictionResult.innerHTML = 'Error making prediction.';
            }
        };
    }
}

async function preprocessImage(img) {
    try {
        // Resize the image (assuming you want 128x128)
        const resized = await resizeImage(img, 128, 128);
        // Convert resized image data to Float32Array
        const float32Array = new Float32Array(128 * 128 * 3);
		
		pixels = resized.data;
		let pixelIndex = 0;
		for (let i = 0; i < float32Array.length; i += 3) {
			float32Array[i] = pixels[pixelIndex]; // Red channel
			float32Array[i + 1] = pixels[pixelIndex + 1]; // Green channel
			float32Array[i + 2] = pixels[pixelIndex + 2]; // Blue channel
			pixelIndex += 4; // Move to the next pixel (4 components: RGBA)
		}

        // Create ONNX Tensor with the correct shape and data type
        const tensor = new ort.Tensor('float32', float32Array, [1, 128, 128, 3]);
        
		return tensor;
    } catch (error) {
        console.error('Error preprocessing image:', error);
        throw error;
    }
}

async function resizeImage(img, width, height) {
    return new Promise((resolve, reject) => {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
		
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, width, height);

        // Get image data from canvas
        const imageData = ctx.getImageData(0, 0, width, height);
        resolve(imageData);
    });
}

// Ensure the functions are available in the global scope
window.predictImage = predictImage;
window.loadModel = loadModel;