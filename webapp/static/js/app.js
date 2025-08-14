class AirplaneDetectionApp {
    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.canvas = null;
        this.ctx = null;
        this.currentImage = null;
        this.startTime = null;
    }

    initializeElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.imageInput = document.getElementById('imageInput');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.resultsSection = document.getElementById('resultsSection');
        this.resultCanvas = document.getElementById('resultCanvas');
        this.errorMessage = document.getElementById('errorMessage');
        this.imageSize = document.getElementById('imageSize');
        this.detectionCount = document.getElementById('detectionCount');
        this.processingTime = document.getElementById('processingTime');
        this.detectionsContainer = document.getElementById('detectionsContainer');
    }

    setupEventListeners() {
        // File upload events
        this.uploadArea.addEventListener('click', () => {
            this.imageInput.click();
        });

        this.imageInput.addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });

        // Drag and drop events
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });

        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                this.handleFileSelect(file);
            }
        });

        // Upload button
        this.uploadBtn.addEventListener('click', () => {
            if (this.currentImage) {
                this.processImage();
            }
        });
    }

    handleFileSelect(file) {
        if (!file) return;

        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file');
            return;
        }

        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            this.showError('File size must be less than 10MB');
            return;
        }

        this.currentImage = file;
        this.updateUploadButton(true);
        this.hideError();
        this.hideResults();

        // Preview the image
        const reader = new FileReader();
        reader.onload = (e) => {
            this.displayPreview(e.target.result);
        };
        reader.readAsDataURL(file);
    }

    displayPreview(imageSrc) {
        const img = new Image();
        img.onload = () => {
            this.setupCanvas(img);
            this.drawImage(img);
        };
        img.src = imageSrc;
    }

    setupCanvas(img) {
        this.canvas = this.resultCanvas;
        this.ctx = this.canvas.getContext('2d');

        // Set canvas size to match image aspect ratio
        const maxWidth = 800;
        const maxHeight = 600;
        let { width, height } = img;

        if (width > maxWidth) {
            height = (height * maxWidth) / width;
            width = maxWidth;
        }

        if (height > maxHeight) {
            width = (width * maxHeight) / height;
            height = maxHeight;
        }

        this.canvas.width = width;
        this.canvas.height = height;
    }

    drawImage(img) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
    }

    async processImage() {
        if (!this.currentImage) return;

        this.startTime = Date.now();
        this.setLoading(true);
        this.hideError();

        const formData = new FormData();
        formData.append('image', this.currentImage);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Prediction failed');
            }

            this.displayResults(result);
        } catch (error) {
            console.error('Prediction error:', error);
            this.showError(error.message || 'Failed to process image');
        } finally {
            this.setLoading(false);
        }
    }

    displayResults(result) {
        const { image, detections, image_size, confidence_breakdown, tile_config } = result;
        const processingTimeMs = Date.now() - this.startTime;

        // Load and display the image
        const img = new Image();
        img.onload = () => {
            this.setupCanvas(img);
            this.drawImage(img);
            this.drawDetections(detections, image_size);
            this.updateResultsInfo(image_size, detections.length, processingTimeMs, confidence_breakdown, tile_config);
            this.displayDetectionsList(detections);
            this.showResults();
        };
        img.src = image;
    }

    drawDetections(detections, originalSize) {
        const scaleX = this.canvas.width / originalSize[0];
        const scaleY = this.canvas.height / originalSize[1];

        detections.forEach((detection, index) => {
            const [x1, y1, x2, y2] = detection.bbox;

            // Scale coordinates to canvas size
            const canvasX1 = x1 * scaleX;
            const canvasY1 = y1 * scaleY;
            const canvasX2 = x2 * scaleX;
            const canvasY2 = y2 * scaleY;

            const width = canvasX2 - canvasX1;
            const height = canvasY2 - canvasY1;

            // Choose color based on confidence level (matching notebook visualization)
            let strokeColor, fillColor, confLevel;
            if (detection.confidence >= 0.7) {
                strokeColor = '#ff4444'; // Red for high confidence
                fillColor = '#ff4444';
                confLevel = 'HIGH';
            } else if (detection.confidence >= 0.5) {
                strokeColor = '#ff8800'; // Orange for medium confidence
                fillColor = '#ff8800';
                confLevel = 'MED';
            } else {
                strokeColor = '#ffdd00'; // Yellow for low confidence
                fillColor = '#ffaa00';
                confLevel = 'LOW';
            }

            // Draw bounding box with confidence-based color
            this.ctx.strokeStyle = strokeColor;
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(canvasX1, canvasY1, width, height);

            // Draw label background
            const label = `${detection.class}: ${(detection.confidence * 100).toFixed(1)}% (${confLevel})`;
            this.ctx.font = '14px Arial';
            const textWidth = this.ctx.measureText(label).width;
            const textHeight = 18;

            this.ctx.fillStyle = fillColor;
            this.ctx.fillRect(canvasX1, canvasY1 - textHeight - 4, textWidth + 8, textHeight + 4);

            // Draw label text
            this.ctx.fillStyle = 'white';
            this.ctx.font = 'bold 14px Arial';
            this.ctx.fillText(label, canvasX1 + 4, canvasY1 - 6);
        });
    }

    displayDetectionsList(detections) {
        this.detectionsContainer.innerHTML = '';

        if (detections.length === 0) {
            this.detectionsContainer.innerHTML = '<p style="text-align: center; color: #666;">No aircraft detected</p>';
            return;
        }

        // Sort detections by confidence (highest first)
        const sortedDetections = [...detections].sort((a, b) => b.confidence - a.confidence);

        sortedDetections.forEach((detection, index) => {
            const detectionElement = document.createElement('div');
            detectionElement.className = 'detection-item';

            const [x1, y1, x2, y2] = detection.bbox;

            // Determine confidence level and color
            let confidenceClass, confidenceLevel;
            if (detection.confidence >= 0.7) {
                confidenceClass = 'confidence-high';
                confidenceLevel = 'HIGH';
            } else if (detection.confidence >= 0.5) {
                confidenceClass = 'confidence-medium';
                confidenceLevel = 'MED';
            } else {
                confidenceClass = 'confidence-low';
                confidenceLevel = 'LOW';
            }

            detectionElement.classList.add(confidenceClass);

            detectionElement.innerHTML = `
                <div class="detection-class">${detection.class} #${index + 1}</div>
                <div class="detection-confidence">
                    <span class="confidence-badge ${confidenceClass}">${confidenceLevel}</span>
                    ${(detection.confidence * 100).toFixed(1)}% confidence
                </div>
                <div class="detection-bbox">
                    Location: (${Math.round(x1)}, ${Math.round(y1)}) → (${Math.round(x2)}, ${Math.round(y2)})
                </div>
                <div class="detection-size">
                    Size: ${Math.round(x2-x1)} × ${Math.round(y2-y1)} pixels
                </div>
            `;

            this.detectionsContainer.appendChild(detectionElement);
        });
    }

    updateResultsInfo(imageSize, detectionCount, processingTime, confidenceBreakdown, tileConfig) {
        this.imageSize.textContent = `${imageSize[0]} × ${imageSize[1]}`;

        // Enhanced detection count with confidence breakdown
        if (confidenceBreakdown) {
            const breakdown = `${detectionCount} total (H:${confidenceBreakdown.high}, M:${confidenceBreakdown.medium}, L:${confidenceBreakdown.low})`;
            this.detectionCount.textContent = breakdown;
            this.detectionCount.title = 'High ≥0.7, Medium 0.5-0.7, Low 0.25-0.5';
        } else {
            this.detectionCount.textContent = detectionCount;
        }

        // Enhanced processing time with inference method
        let timeText = `${processingTime}ms`;
        if (tileConfig) {
            timeText += ` (Tiled: ${tileConfig.tile_size}, overlap: ${tileConfig.overlap}px)`;
        }
        this.processingTime.textContent = timeText;
    }

    updateUploadButton(hasFile) {
        const btnText = this.uploadBtn.querySelector('.btn-text');
        if (hasFile) {
            btnText.textContent = 'Detect Airplanes';
            this.uploadBtn.disabled = false;
        } else {
            btnText.textContent = 'Select Image First';
            this.uploadBtn.disabled = true;
        }
    }

    setLoading(loading) {
        const btnText = this.uploadBtn.querySelector('.btn-text');
        const btnLoader = this.uploadBtn.querySelector('.btn-loader');

        if (loading) {
            btnText.hidden = true;
            btnLoader.hidden = false;
            this.uploadBtn.disabled = true;
        } else {
            btnText.hidden = false;
            btnLoader.hidden = true;
            this.uploadBtn.disabled = false;
        }
    }

    showResults() {
        this.resultsSection.hidden = false;
        this.resultsSection.classList.add('fade-in');
    }

    hideResults() {
        this.resultsSection.hidden = true;
        this.resultsSection.classList.remove('fade-in');
    }

    showError(message) {
        const errorText = this.errorMessage.querySelector('.error-text');
        errorText.textContent = message;
        this.errorMessage.hidden = false;

        // Auto-hide after 5 seconds
        setTimeout(() => {
            this.hideError();
        }, 5000);
    }

    hideError() {
        this.errorMessage.hidden = true;
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AirplaneDetectionApp();
});
